"""Dependency-free tests of the judge pipeline, separate from RAG dependencies."""
import json
import pytest
from judgepipe.schema import Case, load_cases
from judgepipe.rubric import RUBRIC
from judgepipe.parsing import json_object, parse_pairwise, parse_pointwise, consensus, canonical_winner
from judgepipe.runner import Runner, aggregate, kappa
from judgepipe.client import Completion, family


def point(score=4):
    return {'scores': {key: {'score':score,'rationale':'Specific and grounded evidence'} for key in RUBRIC}}


def pair(winner='A', scorea=4, scoreb=1):
    p=point(scorea)['scores']; q=point(scoreb)['scores']
    return {'winner':winner,'rationale':'Answer contains the supported fact',
            'unsupported_length_penalty_applied':False,'scores_a':p,'scores_b':q}


def test_parse_fenced_and_trailing_comma():
    raw=json.dumps(point())
    assert parse_pointwise('```json\n' + raw + '\n```')['overall'] == 4
    assert json_object('intro {"x":2,} trailing')=={'x':2}


def test_parse_legacy_root_level_baseline_scores_without_coercion():
    root_scores = point()['scores']
    assert parse_pointwise(json.dumps(root_scores))['overall'] == 4
    root_scores['correctness']['score'] = 5
    assert parse_pointwise(json.dumps(root_scores)) is None


def test_reject_invalid_and_bool_scores():
    x=point();x['scores']['correctness']['score']=True
    assert parse_pointwise(json.dumps(x)) is None
    x=point(); x['scores']['correctness']['rationale']=''
    assert parse_pointwise(json.dumps(x)) is None
    assert parse_pairwise('{"winner":"A"}') is None


def test_pairwise_and_order_remap():
    assert parse_pairwise(json.dumps(pair()))['winner']=='A'
    assert canonical_winner('B',False)=='A'
    assert canonical_winner('tie',False)=='tie'
    assert consensus('A','A')==('A',False)
    assert consensus('A','B')==('tie',True)
    assert consensus('A',None)==(None,None)


def test_pairwise_boolean_flag_rejects_integer_alias():
    verdict = pair()
    verdict['unsupported_length_penalty_applied'] = 1
    assert parse_pairwise(json.dumps(verdict)) is None


def test_gold_agreement_metrics():
    assert kappa(['A','B','A','B'],['A','B','A','B'])==1
    assert kappa([],[]) is None
    assert family('claude-3-7-sonnet')=='anthropic'
    assert family('gpt-4.1-mini')=='openai'
    assert family('unlisted-model')=='unknown'


def test_kappa_is_undefined_for_constant_labels_and_rejects_mismatched_lengths():
    assert kappa(['A', 'A'], ['A', 'A']) is None
    with pytest.raises(ValueError, match='equally sized'):
        kappa(['A', 'B'], ['A'])


def test_valid_suite_and_case_errors(tmp_path):
    cases=load_cases('judgepipe/data/suite.json')
    assert len(cases)==16
    assert sum(c.probe=='padded' for c in cases)==3
    p=tmp_path/'x.json'
    p.write_text(json.dumps([{'id':'x','input':'why','system_prompt':'answer','model_output':'x'},
                             {'id':'x','input':'what','system_prompt':'answer','model_output':'y'}]))
    with pytest.raises(ValueError,match='duplicate'):
        load_cases(p)


def test_balanced_suite_preserves_references_and_swaps_only_candidate_orientation():
    from collections import Counter
    import hashlib
    from pathlib import Path
    source_path = Path('judgepipe/data/suite.json')
    source = {c.id:c for c in load_cases(source_path)}
    balanced = load_cases('judgepipe/data/suite_balanced.json')
    metadata = json.loads(Path('judgepipe/data/suite_balanced.json').read_text(encoding='utf8'))
    assert metadata['derived_from']['sha256'] == hashlib.sha256(source_path.read_bytes()).hexdigest().upper()
    assert Counter(c.gold_winner for c in balanced) == {'A':7, 'B':7, 'tie':2}
    swapped_probes = set()
    for c in balanced:
        assert c.id.endswith('_balanced')
        original = source[c.id.removesuffix('_balanced')]
        assert (c.input,c.system_prompt,c.expected_output,c.context,c.criteria,c.probe) == (
            original.input,original.system_prompt,original.expected_output,original.context,original.criteria,original.probe)
        if c.gold_winner == 'B':
            assert (c.model_output,c.candidate_b) == (original.candidate_b,original.model_output)
            assert original.gold_winner == 'A'
            swapped_probes.add(c.probe)
        else:
            assert (c.model_output,c.candidate_b,c.gold_winner) == (
                original.model_output,original.candidate_b,original.gold_winner)
    assert {'padded','confidently_wrong'} <= swapped_probes


@pytest.mark.parametrize('gold', ['A', 'B'])
@pytest.mark.parametrize('probe,metric,baseline_metric,count_key', [
    ('padded','padded_answer_wrong_win_rate','padded_answer_wrong_win_rate_baseline','padded_probes'),
    ('confidently_wrong','confidently_wrong_wrong_win_rate','confidently_wrong_wrong_win_rate_baseline','confidently_wrong_probes'),
])
def test_adversarial_wrong_win_metric_is_candidate_orientation_independent(gold, probe, metric, baseline_metric, count_key):
    wrong = 'B' if gold == 'A' else 'A'
    rows = [{'pairwise':True,'gold_winner':gold,'probe':probe,
             'decision':decision,'baseline_winner':decision}
            for decision in (wrong,gold,'tie')]
    rows.append({'pairwise':True,'gold_winner':'tie','probe':probe,
                 'decision':'A','baseline_winner':'A'})
    report = aggregate(rows,mode='offline-test',repeats=2,judge_model='test',generator_model='test')
    assert report['bias'][metric] == 0.3333
    assert report['bias'][baseline_metric] == 0.3333
    assert report['bias'][count_key] == 3


class StableJudge:
    """TEST DOUBLE, NOT A LANGUAGE MODEL. Always prefers original 'GOOD' regardless of slot."""
    def __init__(self, broken_once=False):
        self.calls=0
        self.broken_once=broken_once

    def complete(self,messages):
        self.calls+=1
        if self.broken_once and self.calls==1:
            return Completion('not a JSON response',3,2,'test-double')
        u=messages[1]['content']
        if 'Candidate A (UNTRUSTED)' in u:
            a=u.split('<answer_a>\n',1)[1].split('\n</answer_a>',1)[0]
            b=u.split('<answer_b>\n',1)[1].split('\n</answer_b>',1)[0]
            w='A' if a=='GOOD' else ('B' if b=='GOOD' else 'tie')
            val=pair(w)
            return Completion(json.dumps(val),20,10,'test-double')
        return Completion(json.dumps(point()),20,10,'test-double')


def case():
    return Case('c1','Where?','Answer accurately','GOOD','BAD','GOOD','GOOD','', 'A','ordinary')


def test_live_logic_with_test_double_and_audit(tmp_path):
    judge=StableJudge(broken_once=True)
    run=Runner(judge,judge_model='test-double',generator_model='test-generator',out_dir=tmp_path,
               backend='offline-test',repeats=2)
    result=run.run([case()])
    assert result['report_kind']=='OFFLINE_TEST_DOUBLE_NOT_LLM_EVIDENCE'
    assert result['comparison']['agreement_with_gold']==1
    assert result['bias']['position_order_disagreement_rate_BEFORE_consensus']==0
    assert result['bias']['test_retest_decision_agreement']==1
    assert result['audit']['api_calls']==7  # 1 invalid + 1 retry + 1 pointwise B + 4 pairwise
    assert len((tmp_path/'audit.jsonl').read_text().splitlines())==7
    assert (tmp_path/'report.json').exists()


class PositionBiasedJudge(StableJudge):
    def complete(self,messages):
        self.calls+=1
        if 'Candidate A (UNTRUSTED)' in messages[1]['content']:
            return Completion(json.dumps(pair('A')),20,10,'position-biased-test-double')
        return Completion(json.dumps(point()),20,10,'position-biased-test-double')


def test_position_bias_downgraded_to_tie(tmp_path):
    result=Runner(PositionBiasedJudge(),judge_model='test',generator_model='test',
                  out_dir=tmp_path,backend='offline-test',repeats=2).run([case()])
    assert result['bias']['position_order_disagreement_rate_BEFORE_consensus']==1
    assert result['comparison']['ties']==1
    assert result['bias']['position_sensitive_decisions_after_consensus']==0


def test_baseline_pair_prompt_requires_nested_score_objects():
    from judgepipe.runner import baseline_pair_prompt
    prompt = baseline_pair_prompt(case(), 'answer a', 'answer b')
    assert '"correctness":{"score":0,"rationale":"reason"}' in prompt
    assert 'integer from 0 through 4; never use 5' in prompt
    assert 'Do not flatten criteria into numbers' in prompt


def test_baseline_and_controlled_judges_receive_identical_task_information():
    from judgepipe.runner import case_information, point_prompt, pair_prompt, baseline_point_prompt, baseline_pair_prompt
    c = Case('source', 'QUESTION', 'SYSTEM_CONSTRAINT', 'GOOD', 'BAD',
             'REFERENCE_FACT', 'SOURCE_FACT', ('CUSTOM_CRITERION',), 'A')
    information = case_information(c)
    for prompt in (point_prompt(c, 'GOOD'), pair_prompt(c, 'GOOD', 'BAD'),
                   baseline_point_prompt(c, 'GOOD'), baseline_pair_prompt(c, 'GOOD', 'BAD')):
        assert information in prompt


class InvalidPointwiseBJudge(StableJudge):
    def complete(self, messages):
        if '<answer>\nBAD\n</answer>' in messages[1]['content']:
            return Completion('invalid verdict', 20, 10, 'test')
        return super().complete(messages)


def test_incomplete_when_pointwise_b_is_invalid_even_if_pairwise_succeeds(tmp_path):
    result = Runner(InvalidPointwiseBJudge(), judge_model='test', generator_model='test',
                    out_dir=tmp_path, backend='offline-test').run([case()])
    assert result['comparison']['decisive_or_tie'] == 1
    assert result['pointwise']['config_B_graded'] == 0
    assert result['audit']['invalid_verdicts'] == 1
    assert result['run_status'] == 'incomplete'


def test_incomplete_when_requested_baseline_verdicts_are_invalid(tmp_path):
    class InvalidBaselineJudge(StableJudge):
        def complete(self, messages):
            if messages[0]['content'] in ('You are grading a response. Return requested JSON.',
                                          'You are comparing answers. Return requested JSON.'):
                return Completion('invalid baseline', 20, 10, 'test')
            return super().complete(messages)
    result = Runner(InvalidBaselineJudge(), judge_model='test', generator_model='test',
                    out_dir=tmp_path, backend='offline-test', include_baseline=True).run([case()])
    assert result['comparison']['decisive_or_tie'] == 1
    assert result['audit']['failed_calls'] == 0
    assert result['audit']['invalid_verdicts'] == 3
    assert result['run_status'] == 'incomplete'


def test_interruption_invalidates_previous_complete_report(tmp_path):
    run = Runner(StableJudge(), judge_model='test', generator_model='test',
                 out_dir=tmp_path, backend='offline-test')
    assert run.run([case()])['run_status'] == 'complete'
    run.max_calls = 1
    with pytest.raises(RuntimeError, match='budget'):
        run.run([case()])
    status = json.loads((tmp_path/'report.json').read_text())
    assert status['run_status'] == 'incomplete'
    assert status['failure_type'] == 'RuntimeError'
    assert json.loads((tmp_path/'per_case.json').read_text()) == []
    assert len((tmp_path/'audit.jsonl').read_text().splitlines()) == 1


@pytest.mark.parametrize('reason', ['length', 'max_tokens', 'content_filter', 'tool_calls', 'tool_use', 'pause_turn'])
def test_judge_rejects_parseable_verdict_with_nonfinal_stop_and_preserves_usage(tmp_path, reason):
    class NonFinalJudge:
        def complete(self, messages):
            return Completion(json.dumps(point()), 13, 7, 'test', finish_reason=reason,
                              usage={'input_tokens':13,'output_tokens':7})
    run = Runner(NonFinalJudge(), judge_model='test', generator_model='test',
                 out_dir=tmp_path, backend='offline-test')
    verdict = run._ask('prompt', parse_pointwise, 'c', 'pointwise_A')
    assert verdict['parsed'] is None
    assert verdict['attempts'] == 1
    assert verdict['error'] == f'non_final_completion:{reason}'
    audit = json.loads((tmp_path/'audit.jsonl').read_text())
    assert audit['finish_reason'] == reason
    assert audit['raw_response'] == json.dumps(point())
    assert audit['raw_usage'] == {'input_tokens':13,'output_tokens':7}
    assert verdict['calls'][0]['prompt_tokens'] == 13


def test_empty_judge_completion_rejected_before_parser(tmp_path):
    class EmptyJudge:
        def complete(self, messages):
            return Completion('  ', 13, 7, 'test', finish_reason='end_turn')
    run = Runner(EmptyJudge(), judge_model='test', generator_model='test',
                 out_dir=tmp_path, backend='offline-test')
    verdict = run._ask('prompt', lambda text: {'would_have_passed':True}, 'c', 'pointwise_A')
    assert verdict['parsed'] is None
    assert verdict['error'] == 'empty_completion'


@pytest.mark.parametrize('text,reason', [('GOOD','max_tokens'), ('','end_turn')])
def test_generator_rejects_nonfinal_or_empty_answer_with_recorded_audit(tmp_path, text, reason):
    class InvalidGenerator:
        def complete(self, messages):
            return Completion(text, 17, 9, 'generator-test', finish_reason=reason,
                              usage={'input_tokens':17,'output_tokens':9})
    c = Case('generated', 'Q', 'Answer from context', '', context='source')
    run = Runner(StableJudge(), generator=InvalidGenerator(), judge_model='test',
                 generator_model='generator-test', out_dir=tmp_path,
                 backend='offline-test', generate_missing=True)
    with pytest.raises(RuntimeError, match='Generator completion rejected'):
        run.run([c])
    assert json.loads((tmp_path/'report.json').read_text())['run_status'] == 'incomplete'
    audit = json.loads((tmp_path/'audit.jsonl').read_text())
    assert audit['task'] == 'generator'
    assert audit['finish_reason'] == reason
    assert audit['raw_usage']['output_tokens'] == 9
    assert audit['error']


class ChangingRawJudge(StableJudge):
    def __init__(self):
        super().__init__()
        self.pair_calls = 0

    def complete(self, messages):
        if 'Candidate A (UNTRUSTED)' in messages[1]['content']:
            self.pair_calls += 1
            winner = 'A' if self.pair_calls <= 2 else 'B'
            return Completion(json.dumps(pair(winner)), 20, 10, 'test')
        return super().complete(messages)


def test_raw_same_order_variation_is_not_hidden_by_repeated_consensus_ties(tmp_path):
    result = Runner(ChangingRawJudge(), judge_model='test', generator_model='test',
                    out_dir=tmp_path, backend='offline-test', repeats=2).run([case()])
    assert result['bias']['test_retest_decision_agreement'] == 1
    assert result['bias']['raw_same_order_retest_disagreement_rate'] == 1
    assert result['bias']['raw_same_order_retest_comparisons'] == 2
    assert result['bias']['position_reverse_trials_all_repeats'] == 2


def test_all_repeat_swap_metric_includes_later_disagreement():
    rows = [{'pairwise':True, 'decision':'tie', 'position_flip':False,
             'first_order_winner':'A', 'retest_consistent':False,
             'pairwise_repeat_1':{'AB_canonical_winner':'A', 'BA_canonical_winner':'A', 'position_flip':False},
             'pairwise_repeat_2':{'AB_canonical_winner':'A', 'BA_canonical_winner':'B', 'position_flip':True}}]
    report = aggregate(rows, mode='offline-test', repeats=2, judge_model='test', generator_model='test')
    assert report['bias']['position_order_disagreement_rate_BEFORE_consensus'] == 0
    assert report['bias']['position_order_disagreement_rate_all_repeats'] == 0.5


def test_generator_gets_context_and_criteria_without_gold_and_usage_stays_separate(tmp_path):
    class Generator:
        def complete(self, messages):
            self.messages = messages
            return Completion('GOOD', 17, 9, 'generator-test')

    generator = Generator()
    c = Case('generated', 'QUESTION', 'GROUNDING_RULE', '', 'BAD',
             'HIDDEN_GOLD_REFERENCE', 'VISIBLE_SOURCE', ('VISIBLE_CRITERION',), 'A')
    result = Runner(StableJudge(), generator=generator, judge_model='test',
                    generator_model='generator-test', out_dir=tmp_path,
                    backend='offline-test', generate_missing=True).run([c])
    messages = json.dumps(generator.messages)
    assert 'VISIBLE_SOURCE' in messages and 'VISIBLE_CRITERION' in messages
    assert 'HIDDEN_GOLD_REFERENCE' not in messages
    assert result['generation_usage']['prompt_tokens'] == 17
    assert result['generation_usage']['completion_tokens'] == 9
    assert result['generation_usage']['api_calls'] == 1
    assert result['audit']['api_calls'] == 6
    assert result['audit']['prompt_tokens'] == 120
    assert len((tmp_path/'audit.jsonl').read_text().splitlines()) == 7


def test_consolidation_rejects_stale_originals_after_new_run(tmp_path, monkeypatch):
    from scripts.problem2_consolidate_live_results import main
    primary = tmp_path/'primary'
    primary.mkdir()
    (primary/'report_original.json').write_text(json.dumps({'generated_at_utc':'old-run'}))
    (primary/'report.json').write_text(json.dumps({'generated_at_utc':'new-run'}))
    monkeypatch.setattr('sys.argv', ['consolidate', '--primary', str(primary),
                                    '--supplement', str(tmp_path/'supplement')])
    with pytest.raises(ValueError, match='stale originals'):
        main()


def test_native_anthropic_request_and_token_usage(monkeypatch):
    from judgepipe import client as lib
    from io import BytesIO

    captured = {}
    class FakeResponse:
        def __enter__(self):
            return BytesIO(json.dumps({'model':'claude-test',
                'content':[{'type':'text','text':'{"ok":true}'}],
                'usage':{'input_tokens':32,'output_tokens':11}}).encode())
        def __exit__(self, *args):
            pass

    def fake_urlopen(request, timeout):
        captured['request'] = request
        captured['timeout'] = timeout
        return FakeResponse()

    monkeypatch.setattr(lib, 'urlopen', fake_urlopen)
    cfg = lib.ClientConfig('https://api.anthropic.com/v1','claude-test','fake-test-token',provider='anthropic')
    response = lib.make_client(cfg).complete([
        {'role':'system','content':'Follow rubric'},
        {'role':'user','content':'Grade answer'}])
    assert response.text == '{"ok":true}'
    assert response.prompt_tokens == 32
    assert response.completion_tokens == 11
    assert captured['request'].full_url == 'https://api.anthropic.com/v1/messages'
    payload = json.loads(captured['request'].data)
    assert payload['system'] == 'Follow rubric'
    assert payload['messages'] == [{'role':'user','content':'Grade answer'}]
    assert 'temperature' not in payload
    assert 'system' not in str(payload['messages'])
    assert captured['request'].get_header('X-api-key') == 'fake-test-token'
    assert captured['request'].get_header('Anthropic-version') == '2023-06-01'


def test_anthropic_environment_and_errors(monkeypatch):
    from judgepipe.client import ClientConfig, make_client
    monkeypatch.setenv('JUDGE_PROVIDER','anthropic')
    monkeypatch.setenv('JUDGE_MODEL','claude-example')
    monkeypatch.setenv('ANTHROPIC_API_KEY','fake-test-token')
    monkeypatch.delenv('JUDGE_API_KEY',raising=False)
    cfg = ClientConfig.from_env('JUDGE')
    assert cfg.provider == 'anthropic'
    assert cfg.base_url == 'https://api.anthropic.com/v1'
    assert cfg.api_key == 'fake-test-token'
    assert cfg.temperature is None
    assert make_client(cfg).__class__.__name__ == 'AnthropicClient'


def test_anthropic_temperature_can_be_explicitly_enabled(monkeypatch):
    from judgepipe.client import ClientConfig
    monkeypatch.setenv('JUDGE_PROVIDER','anthropic')
    monkeypatch.setenv('JUDGE_TEMPERATURE','0.25')
    cfg = ClientConfig.from_env('JUDGE')
    assert cfg.temperature == 0.25


def test_submission_summary_refuses_mock_results():
    from scripts.problem2_submission_summary import report_markdown
    with pytest.raises(ValueError, match='test-double'):
        report_markdown({'report_kind':'OFFLINE_TEST_DOUBLE_NOT_LLM_EVIDENCE'})


def test_submission_summary_rejects_incomplete_live_report():
    from scripts.problem2_submission_summary import report_markdown
    with pytest.raises(ValueError, match='Incomplete'):
        report_markdown({'report_kind':'LIVE_LLM_JUDGE', 'run_status':'incomplete'})

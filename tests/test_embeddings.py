import json

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

from docqa.embeddings import audit_tokenizer_inputs


def test_truncation_audit_counts_real_model_tokens_and_preserves_model_configuration():
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "word": 3}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.post_processor = TemplateProcessing(single="[CLS] $A [SEP]",
                                                  special_tokens=[("[CLS]", 1), ("[SEP]", 2)])
    tokenizer.enable_truncation(max_length=6)
    tokenizer.enable_padding(length=6)
    before = json.loads(tokenizer.to_str())
    report = audit_tokenizer_inputs(tokenizer, ["word word", "word " * 8], "test-model")
    assert report["available"]
    assert report["max_length"] == 6
    assert report["n_inputs"] == 2 and report["n_truncated"] == 1
    assert report["max_input_tokens"] == 10 and report["tokens_discarded"] == 4
    assert report["per_input"][0]["tokens_including_special"] == 4
    assert json.loads(tokenizer.to_str()) == before
    assert len(tokenizer.encode("word " * 8).ids) == 6


def test_truncation_audit_is_explicit_when_backend_has_no_tokenizer():
    report = audit_tokenizer_inputs(None, ["text"], "test-model")
    assert not report["available"] and report["reason"]

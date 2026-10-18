# Submission reflection answers

All six reflection answers are included in the submission document. The Problem 1 answers were supplied by the candidate; the AI disclosures describe assistance with the work and final preparation. The original assignment instructions remain preserved in `submission/source_material/`.

## Problem 1 — walk through one query

Say someone asks, "What is the maximum time to respond to a code review request?"

First, the question is turned into a vector using the same embedding model I used for the documents, BAAI/bge-small-en-v1.5. Then I run two searches on the same SQLite database: a vector search using cosine similarity and a keyword search using BM25. Each search returns its top 20 chunks.

Before the keyword search, I clean up the question. I keep only word tokens, remove common words like "what" and "the", and put the remaining words in quotes. This also prevents special characters from interfering with the full-text search query. Filters like collection and document type are only accepted from a fixed list, because their column names are used directly in SQL.

Next, I combine the two result lists using reciprocal rank fusion (RRF). Each chunk gets a score of 1/(60 + its rank) from each search where it appears. I add those scores together and keep the top 5 chunks. I use ranks because BM25 scores and cosine similarity scores aren't on the same scale.

I actually added keyword search after my evaluation showed that vector search alone couldn't find the evidence for one question, q21, which was about secrets. The relevant evidence wasn't appearing even in the top 10 results. After combining both searches, Recall@5 improved from 0.96 to 1.0, and MRR@5 increased from 0.75 to about 0.87. In that original evaluation, the additional keyword search added less than a millisecond to the median search time.

After retrieval, there's a similarity gate. I take the highest cosine similarity score from the vector search. If it's below 0.5, the system directly responds with "I don't know based on the provided documents." It doesn't call the LLM, so no generation tokens are used. I chose the cosine score rather than the fused RRF score because the RRF values are small and based on ranking positions, making them unsuitable for the same similarity threshold.

For this question, the highest similarity was 0.817, so it passed the gate.

The top 5 chunks are then added to the prompt, numbered from 1 to 5. Each passage includes its title, document reference, section, and page number where available. The LLM is instructed to answer using only those passages and include citation numbers after every sentence. If the answer isn't supported by the documents, it must return the exact "I don't know" response.

For this particular query, the LLM used 2,336 input tokens and 66 output tokens. The model call took 778 ms.

Finally, I check the citations returned by the model. The system reads citation markers such as [1] and [3] and maps them back to the retrieved chunks. If a citation number is outside the available range, it's recorded as invalid. If an answer contains no valid citations, it's flagged in the logs for review.

The final answer said that one business day is the maximum time to respond to a code review request. It also mentioned that reviewers should wait for a natural break rather than interrupt focused work. The answer cited [1] and [3], both from the *Speed of Code Reviews* document.

The entire query took about 795 ms, including retrieval and generation. The system also logged the execution time, token usage, similarity score, and citation details.

## Problem 1 — something that broke the first time

One issue I ran into was with the "I don't know" check. The code was running without errors, so initially I thought it was working correctly. I only noticed the problem when I ran the evaluation.

At first, I had set the similarity threshold to 0.35. The idea was simple: if the closest document chunk had a similarity score below 0.35, the system should refuse to answer without calling the LLM.

But during the initial evaluation, it didn't block any of the 5 unanswerable questions.

For example, I tested the question "Who won the 2018 FIFA World Cup?" There was nothing about football in my documents, but the embedding model still gave it a similarity score of approximately 0.44. Since that was above 0.35, the question passed the gate and went to the LLM.

The model still responded that it didn't know, because of the instructions in the prompt. So the final answer looked correct, but the system was making unnecessary LLM calls.

When I checked the similarity scores, I found that the embedding model was giving relatively high scores even to questions that weren't answerable from the documents. All 25 answerable questions had scores of at least 0.682, which meant my original threshold of 0.35 was too low to be useful.

There was another interesting problem with questions that were related to the documents but didn't have an answer in them.

For example, "Which port does the Prometheus Pushgateway listen on by default?" was unanswerable using my documents, but its similarity score was 0.7222. At the same time, one answerable question had a score of 0.7221.

That showed me that a single similarity threshold couldn't perfectly separate answerable and unanswerable questions.

I then tested different threshold values using the evaluation set. I eventually chose 0.5.

With this threshold, the system could reject clearly unrelated questions while still keeping all 25 answerable questions. I also tested 0.65, which rejected more unanswerable questions, but I didn't choose it because the lowest answerable similarity was only 0.682. With such a small evaluation set, I felt that setting the threshold so close to the answerable scores could cause problems with new questions.

I left the more difficult cases to the LLM, with instructions to answer only when the retrieved passages actually contained the required information.

After changing the threshold, all 5 unanswerable questions were correctly declined. One was rejected by the similarity gate without an LLM call, while the other four were declined by the model. None of the 25 answerable questions was incorrectly rejected.

The main thing I learned was that I shouldn't choose a similarity threshold just because the number looks reasonable. Its behaviour depends on the embedding model and the documents being searched, so it needs to be tested using actual questions.

I also learned that a system can produce the correct final answer while still having a problem internally. Without evaluating the similarity gate separately, I probably wouldn't have noticed the unnecessary LLM calls.

## Problem 1 — AI disclosure

I used AI mainly when I needed help understanding some technical concepts, fixing errors, and making my explanations clearer. I also provided my own pseudocode to guide how I wanted the solution to work, rather than depending completely on AI-generated suggestions.

I tested the retrieval pipeline, compared different similarity thresholds, and checked the answers against the actual documents. I didn't just accept the AI output as it was. I went through the results myself and made the final decisions based on what the testing showed.

I also used Codex to place these answers in the final submission and check the document formatting and consistency with the saved results.

## Problem 2 — position-bias check

I checked whether changing the order of the answers would change the judge’s choice. Each pair was shown as A–B and then B–A, and the verdicts were compared using the original answer identities. I also repeated the judgments with the order unchanged to check how consistent they were.

The simpler baseline changed its choice on one of the 16 pairs. Both answers in that case said essentially the same thing about environment variables, so there was little reason to prefer either one. The rubric-based judge returned a tie. Across its two repetitions, all 32 order comparisons agreed, and the unchanged-order checks were stable too.

The results are encouraging, but I wouldn’t say position bias is solved. These are only 16 selected examples. Also, the pipeline turns conflicting verdicts into ties as a precaution; the zero disagreement reported here was measured before that rule was applied.

## Problem 2 — adversarial-result interpretation

I wanted to see whether an answer could win just by being longer or sounding more certain. Across the three padded-answer cases and five confidently wrong cases, none of those answers beat the reference-preferred response. The check followed the actual answer through both presentation orders, so changing its A/B label did not change what counted as a wrong choice.

For example, one answer confidently claimed that Pushgateway should be used for every application. The source recommends it only in limited circumstances, and the judge chose the answer that respected that restriction in both orders. In another case, both answers correctly mentioned environment variables, but one added generic advice that did not help answer the question. The judge preferred the direct response.

I would be careful about interpreting that second example. A longer answer is not necessarily worse, and confidence is not necessarily a warning sign. What matters is whether the extra detail is accurate and useful. A stronger test would include a longer answer with necessary qualifications, a short answer that leaves out something important, and an incorrect answer written in a cautious tone. That would help check whether the judge is assessing the content rather than learning a shortcut such as “shorter is better.”

The baseline passed all eight probes too. These results therefore do not show that the additional controls improved performance on these cases. They show that both versions handled the selected examples correctly.

The examples also have limits. Some errors, such as recommending Pushgateway for every application “without exceptions,” are fairly obvious once the reference is available. More difficult cases would contain a small factual error inside an otherwise convincing answer, an omitted exception, or a correct statement attributed to the wrong source. The expected labels here were derived from references rather than independent blind human judgments. Before trusting the judge on unfamiliar tasks, I would want to see how it handles those closer decisions and where its reasoning disagrees with human reviewers.

## Problem 2 — AI disclosure

AI helped with most of the implementation, debugging, experiment setup and initial writing. My part was the critical review: checking disputed verdicts against the source material, questioning whether the tests measured what they claimed to measure, and deciding which conclusions the results supported. AI also suggested interpretations and helped with the final wording. I treated those suggestions as proposals to check, rather than accepting them automatically. My responsibility was to verify the reasoning behind the claims I kept.

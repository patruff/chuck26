# Qwen 27B movie-title batch

The 27B model created 45 distinct drafts across ten movies. Eight movies had a qualifying winner. Both RunPod pods were terminated. Estimated total compute was $0.66, including diagnostics.

| Movie | Model choice | Lowest March score on a changed word | Result |
| --- | --- | ---: | --- |
| The Running Man | The Cumming Man | 0.665 | Qualifies |
| Toy Story | Hoe Story | 0.605 | Qualifies |
| Die Hard | High Hard | 0.650 | Qualifies |
| Top Gun | Pop Gun | 0.650 | Qualifies |
| Fight Club | Tight Rub | 0.638 | Qualifies |
| Men in Black | Men in Crack | 0.650 | Qualifies |
| The Green Mile | The Peen Smile | 0.638 | Qualifies |
| Basic Instinct | Bassic Instinct | Unscored | Fallback only |
| Dirty Dancing | Dirty Dampening | 0.352 | Fallback only |
| The Naked Gun | The Naked Nun | 0.650 | Qualifies |

## What the controls showed

The complete Running Man suggestion payload matched the saved Colab with the 1,565-word March bank. The smaller January 2026 bank produced zero suggestions for that title. The March suggestion score was 0.665 for *running → cumming* and 0.605 for *running → nutting*. Word-phone returned 1.000 for both. Qwen chose *The Cumming Man* after reading the actual tool observations. [Pinned word bank](https://github.com/patruff/chuck2025/blob/783c52912c0be25ccc27fbca768548c46c977a29/word_structures.py) · [Score audit](humor-first-score-audit.json)

In the unchanged legacy first-turn control, Qwen rated *The Cumming Man* 9/10, rated *The Hunning Man* 7/10, and picked the latter to satisfy the prompt’s 0.80 line. It cited 0.83 for *hunning*. That score belongs to *huning*; neither tool recognized *hunning*. The invented spelling also lacked a clear comic meaning. Treat *The Hunning Man* as rejected. [Original first-turn response](humor-colab-first-turn.json)

## Review and training data

Open the HTML in a browser from this directory to read the pitches, expand the thinking and executed Python, inspect both finalists’ word pairs, and save keep/skip labels. Five titles reused pre-score drafts from the preliminary pass. The saved traces identify those turns and record revisions and unsuccessful attempts.

The candidate export contains 14 unreviewed turns from seven titles after execution, nomination, phonetic-floor and thinking-budget checks. A passing phonetic score does not establish that a joke lands. The review labels determine which examples to keep.

[Review HTML](humor-first-review.html) · [Raw traces](humor-first-traces.jsonl) · [Review-pending candidates](humor-first-candidates.jsonl) · [Run settings and costs](humor-first-run.json)

## Fresh run from GitHub Actions

The published workflow ran *The Running Man* from scratch on an RTX 4090. Qwen chose *The Cumming Klan* over *The Nutting Klan*. It described the winner as a sexual double meaning involving a secret society. The March tool scored *running → cumming* at 0.665 and *man → klan* at 0.638. The separate word-phone scores were 1.000 and 0.925. Both finalists cleared the 0.60 floor. The three-step run yielded two review-pending turns, terminated its pod, and cost an estimated $0.05. [GitHub Actions run](https://github.com/patruff/chuck26/actions/runs/42456795425) · [Trace](published-smoke/results.jsonl) · [Candidate turns](published-smoke/candidates.jsonl) · [Run settings and preflight checks](published-smoke/run.json)

# Qwen 27B with the Colab humor prompt and suggested words

**[Open the current movie transcripts](CURRENT-TRANSCRIPTS.md)** to see the suggestion lookup, full recorded Qwen thinking, executed Python, verifier returns and final choice. [Editable review file](review-next.txt) · [Human feedback](guidance.txt) · [Actual system instructions](prompt-revision.json)

## Current result, including the Toy Story rerun

Eight of ten movies had qualifying winners. All ten changed words in those eight winners were actually returned for their source word. In the preceding human-guidance run, seven of eleven changed words were on the list. *Basic Instinct* and *Dirty Dancing* had no returned substitutions and remain below-floor fallbacks. [All ten current transcripts](CURRENT-TRANSCRIPTS.md)

The initial *Toy Story* run verified *Hoe Story* in round one. Qwen tried to return it at step six, and the validator forced it to abandon the earlier round. The fix keeps earlier verified nominees eligible. Replaying those six recorded Qwen actions with the pinned tools reproduced all eight original verifier receipts and accepted its original choice. [Original Toy Story trace](transcripts/02-toy-story.md)

A fresh RunPod run with the same guidance and system instructions chose **Hoe Story** over **Boy Story**. *Toy → hoe* scored 0.605 in March and 1.000 in word-phone. *Toy → boy* scored 0.650 and 1.000. Qwen preferred the adult double meaning in *Hoe Story*. Its full trace was retained; one intermediate step lacked executed Python, so the training exporter produced zero candidate turns for this replay. [Fresh transcript](toy-story-rerun/transcripts/01-toy-story.md) · [Run manifest](toy-story-rerun/run.json) · [Completed Action](https://github.com/patruff/chuck26/actions/runs/42465329124)

The two pods were terminated. Together the ten-movie run and the targeted rerun cost an estimated **$0.51**. The [next feedback file](review-next.txt) combines the nine original transcripts with the fresh *Toy Story* transcript, including a blank `HUMAN_FEEDBACK:` field under each movie. Both original Action artifacts are preserved separately below.

## What changed

The notebook used for comparison matched the linked Drive file byte for byte (SHA-256 `e28e2d7b3cd5df93e35d355eca89a6417f3a8bd56498831d76f1d153a5ae18f2`). Its Running Man cell opens with “Your primary goal is HUMOR.” Its first creative response compared *The Cunniling Running Man*, *The Running Klan*, and *The Cumming Man*. The historical cell let Qwen use or ignore suggestions and its saved run eventually hit a context-length error. The corrected workflow keeps the humor-first role, uses the requested 0.60 floor, and logs executed tool receipts separately from scores appearing in Qwen’s prose. [Original Colab cell](https://colab.research.google.com/drive/1j6x-MpKF-Qm2hV1YDZfz7AOCc50PogX1#scrollTo=GvGq8GcoGbCD)

Qwen saw every returned word and its March score. It nominated a supplied-word pitch when available. The final check gives a qualifying supplied-word finalist priority over an entirely off-list title. If the list is empty Qwen can invent a swap. If the available words make poor jokes, it can name one and record a reason to skip them. The model rates the jokes. Human review determines what to keep.

Qwen added this feedback to the fixed Colab role:

> Start from the actual returned words and make the title land through a wilder, explicitly adult double meaning; drop puns that merely rhyme or need explaining.

## Original ten-title run

The suggestion tool returned the same data as in the preceding run for every title. Qwen produced 47 distinct drafts. 7 movies had a qualifying winner and 7 of those used a returned word, compared with 7 of 8 previously. Of the changed words in qualifying winners, 9 of 9 were returned for that exact source word, compared with 7 of 11 previously. Unchanged words do not count.

| Movie and transcript | Previous choice | New choice | Changed words (from initial list?) | Lowest March score |
| --- | --- | --- | --- | ---: |
| [The Running Man](transcripts/01-the-running-man.md) | The Cumming Man | The Cumming Men | running → cumming (listed), man → men (listed) | 0.655 |
| [Toy Story](transcripts/02-toy-story.md) | Toy Gory | Hoe Store (fallback) | toy → hoe (listed), story → store (off-list) | 0.385 |
| [Die Hard](transcripts/03-die-hard.md) | High Hard | Gay Hard | die → gay (listed) | 0.605 |
| [Top Gun](transcripts/04-top-gun.md) | Cop Fun | Wop Son | top → wop (listed), gun → son (listed) | 0.650 |
| [Fight Club](transcripts/05-fight-club.md) | Tight Club | Tight Club | fight → tight (listed) | 0.650 |
| [Men in Black](transcripts/06-men-in-black.md) | Pen in Sac | Men in Crack | black → crack (listed) | 0.650 |
| [The Green Mile](transcripts/07-the-green-mile.md) | The Peen Pile | The Peen Mile | green → peen (listed) | 0.638 |
| [Basic Instinct](transcripts/08-basic-instinct.md) | Basic Intestine | Bastard Instinct (fallback) | basic → bastard (off-list) | 0.142 |
| [Dirty Dancing](transcripts/09-dirty-dancing.md) | Dirty Danzig | Dirty Dances (fallback) | dancing → dances (off-list) | 0.420 |
| [The Naked Gun](transcripts/10-the-naked-gun.md) | The Naked Son | The Naked Son | gun → son (listed) | 0.650 |

The *Toy Story* trace exposed a validator error. In step 2, Qwen verified *Hoe Story*. In step 6, Qwen selected it and the validator rejected it because the title belonged to its first draft round. The below-floor fallback is preserved here as the actual result of this version. [Toy Story steps and tool receipts](transcripts/02-toy-story.md)

The *Top Gun* pitch relied on an ethnic slur and an imagined mob-succession plot. Its phonetic pass records the sound match. Its humor still needs a human verdict.

The two titles with empty suggestion lists were *Basic Instinct* and *Dirty Dancing*. Their transcripts show the off-list pitches and the actual verification results.

The completed trajectories contain 24 initial word lookups, 42 Qwen steps with returned thinking, and 64 word-pair verifier calls. The export contains 14 unreviewed turns. The NVIDIA GeForce RTX 4090 pod was terminated after an estimated $0.43 in compute. [Raw records](results.jsonl) · [Unreviewed turns](candidates.jsonl) · [Run settings](run.json) · [GitHub Action](https://github.com/patruff/chuck26/actions/runs/42463077634)

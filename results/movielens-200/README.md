# MovieLens film preference batch

**185 of 200 films are ready to review.** Each has two distinct parody titles, their pitches, and independent Qwen trajectories. [Browse the catalog](CATALOG.md) · [Download the pair index](pairs.jsonl) · [Open the review app](../../README.md#review-film-parodies-and-export-dpo).

The ranking uses the most-rated films released from 1990 through 2015 inclusive in the MovieLens 32M snapshot. The [film list](../../pipeline/film_lists/top200_1990_2015.csv) preserves the original titles, years, IDs, counts and ranks. [Source and ranking method](../../pipeline/film_lists/README.md).

## Generation and phonetics

The 370 completed smolagents trajectories contain 2,040 recorded steps with Qwen thinking, 3,181 actual phonetic verification calls and 2,278 distinct title drafts counted within their respective runs. Expand either trace in the app to see the original suggestions, the code Qwen executed, tool returns, later revisions and selection.

The suggestion score from the pinned March tool determines whether each changed word clears 0.60. The separate word-phone score is shown alongside it. Both options clear the floor for **86 films**. At least one clears it for **112 films**. The app identifies below-floor alternatives. Where independent runs picked the same title, 67 pairs show a different already-verified finalist from one of the two runs. Each source is recorded in the index and can be opened in the app.

Each A/B pair uses the same guidance on both sides. 95 films use the [human-feedback revision](../feedback-0c3cc72/README.md), 80 the [adult-premise revision](../film-adult-premises/README.md), and 10 the [workplace revision](../film-workplace-premises/README.md). The latter revisions direct the humor toward adult characters or misconduct by adults. [Per-rank prompt versions and batch metadata](batch.json).

RunPod ran out of credit. All 15 allocated pods have confirmed teardown, and the estimated cost including retries and superseded runs was **$16.36**. Shared-server samples are included in their parent allocation. [Run manifests](runs).

## Resume the last 15 films

The [remaining film list](remaining.json) records each target, prompt snapshot, completed side, and any interrupted trajectory with tool receipts. Seven completed sides have been saved, plus six partial trajectories. They stay outside the review queue until both sides complete. After adding RunPod credit, dispatch these two targeted jobs from a checkout with access to the repository’s RunPod and word-bank secrets.

```sh
gh workflow run film-review-batch.yml -R patruff/chuck26 -f shards='[0]' -f use_saved_sides=true -f ranks='[101,135,155,167,175,183]' -f prompt_snapshot=results/film-adult-premises
gh workflow run film-review-batch.yml -R patruff/chuck26 -f shards='[0]' -f use_saved_sides=true -f ranks='[14,39,53,65,79,108,149,156,171]' -f prompt_snapshot=results/film-workplace-premises
```

Each run uploads a `film-shard-0` artifact. Inspect the new pairs and add their `samples/NNN-a.json.gz` and `samples/NNN-b.json.gz` to this batch. Once all 200 are available, regenerate the catalog with `python scripts/film_catalog.py results/movielens-200`.

## Record preferences

Choose **Prefer A**, **Prefer B** or **Neither works**, enter a comment, and save. Every A/B preference creates a conversational row in `reviews/reviewed_dpo.jsonl` containing the shared title prompt, chosen title, rejected title, human feedback, and the source path and SHA-256 for each trajectory. **Neither works** saves the comment in `reviews/film_preferences.jsonl` without creating a DPO row. Download either file in the app. [App setup and trainer command](../../README.md#review-film-parodies-and-export-dpo).

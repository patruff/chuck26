# MovieLens film preference batch

This checkpoint contains **48 reviewable films** from the 200 most-rated MovieLens releases in 1990–2015. Each movie has two independently sampled Qwen trajectories in `samples/`. Open them with `python scripts/review_films.py` from the repo root. The remaining shards and targeted retries are in progress.

The full slate uses the fixed [ranked film list](../../pipeline/film_lists/top200_1990_2015.csv) and the [latest reviewed guidance snapshot](../feedback-0c3cc72/README.md). Each side contains the complete recorded thinking, Python actions, actual verifier returns, finalists, and the model’s selection. The app saves your own preferences in `reviews/reviewed_dpo.jsonl` and keeps comments on rejected pairs in `reviews/film_preferences.jsonl`.

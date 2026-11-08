# MovieLens film preference batch

This in-progress checkpoint contains **95 completed film pairs** from the 200 most-rated MovieLens releases in 1990–2015. Each movie has two independently sampled Qwen trajectories in `samples/`. Open them with `python scripts/review_films.py` from the repo root. The remaining shards and targeted replacements are in progress. The app loads only pairs with two different, completed trajectories.

The full slate uses the fixed [ranked film list](../../pipeline/film_lists/top200_1990_2015.csv) and the [latest reviewed guidance snapshot](../feedback-0c3cc72/README.md). Each side contains the complete recorded thinking, Python actions, actual verifier returns, finalists, and the model’s selection. The app saves your own preferences in `reviews/reviewed_dpo.jsonl` and keeps comments on rejected pairs in `reviews/film_preferences.jsonl`.

Some movies are being regenerated with the [adult-premise revision](../film-adult-premises/README.md). Their replacement traces use the same film list, Qwen model, pinned phonetic tools, and floor.

A small set of additional replacements uses an [adult workplace premise](../film-workplace-premises/README.md), focusing the joke on conduct or a cover-up. Each movie’s two sides use the same revision.

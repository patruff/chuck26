# Chuckles

Chuckles writes movie-title parodies for an adult audience. The working cycle is **generate → judge and comment → export DPO preferences → train an adapter → compare again**.

**[Open the inference form](https://github.com/patruff/chuck26/actions/workflows/human-guidance.yml)** and click **Run workflow**. The **system_prompt** field is prefilled with the exact audience instructions and judged examples. Edit it for this run. For a lasting change, edit the field’s default in [the kickoff workflow](.github/workflows/human-guidance.yml). Local runs read that same default. Each new run saves its prompt and prints it in the Actions summary.

The model is **0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF**, using `RVN-Q4_K_M-multilingual.gguf`. The revision, weight checksum, llama.cpp commit and RunPod image are pinned together in [config/model.json](config/model.json). A DPO run adds a LoRA to these weights.

## GitHub Actions

| Action | When to use it | What comes back |
| --- | --- | --- |
| [1. Generate parodies for review](https://github.com/patruff/chuck26/actions/workflows/human-guidance.yml) | Pick a [movie rank](pipeline/film_lists/top200_1990_2015.csv) and edit the audience prompt. **base-cpu** generates with the original model on a GitHub runner. **runpod** runs the smolagents CodeAgent on a rented GPU. **compare-adapter** runs both model configurations on a held-out movie. **preview** checks the prompt, bank and historical scores without renting a GPU. | Exact prompt plus a `review-ready` artifact for base or RunPod inference, or a recipient-encrypted `sealed-review` for comparison. |
| [2. Train a DPO adapter from reviews](https://github.com/patruff/chuck26/actions/workflows/dpo-smoke.yml) | Supply the key ID from the review-export command below. This trains the tested rank-4 output-layer adapter on your actual A/B preferences and reloads it in llama.cpp. | Encrypted adapter and comparison report. The training run number and key ID identify the adapter in the next inference form. |
| [3. RunPod bulk generation and resume](https://github.com/patruff/chuck26/actions/workflows/film-review-batch.yml) | Generate two independent CodeAgent trajectories per movie, or resume an interrupted shard. Leave **prompt_snapshot** blank for the current prompt. Use **shards** `[0]` with **ranks** such as `[35,45]` for a targeted batch. | Per-side compressed checkpoints. Download a shard and point the app at its `samples/` directory. |

## Review film parodies and export DPO

From this checkout on macOS or Linux with Python 3.10+ and authenticated `gh`:

```sh
python -m pip install -e '.[trace-export]'
python scripts/review_cycle.py fetch INFERENCE_RUN
python scripts/review_films.py
```

Open [the review app](http://127.0.0.1:5118). New runs appear as `inference-RUN`; the app also reads the [185-film catalog](results/movielens-200/CATALOG.md) and earlier feedback results. For a downloaded artifact, use `python scripts/review_films.py --input /path/to/artifact`. The original blind 15-film collection loads from `reviews/blind-27b-15/`.

Read both pitches and expand their reasoning and phonetic checks. Save **Prefer A**, **Prefer B** or **Neither works**, with an optional comment. Blind comparisons also allow **Equally funny**; attribution can be revealed for that film after voting. The app writes all decisions to `reviews/film_preferences.jsonl`. Each A/B choice between distinct titles produces a row in `reviews/reviewed_dpo.jsonl` with the shared prompt, chosen title, rejected title, feedback and exact trace references. **Download DPO** exports that file. Use **Transfer reviews** to import a prior export or backup.

## Train and judge the next adapter

```sh
python scripts/review_cycle.py validate "/path/to/reviewed_dpo.jsonl"
python scripts/review_cycle.py train "/path/to/reviewed_dpo.jsonl" --key .private/dpo.pem
```

The second command creates or reuses your local recipient key, prints a **key ID**, and starts training. Use `--prepare-only` to set up the private request and launch Action 2 yourself. The request sends the actual prompt and title pairs; reviewer notes and tool traces stay in the full export. For a large export you deliberately committed to `reviews/` on GitHub main, use `--tracked` with its local path. [DPO training and model details](training/gguf_dpo/README.md)

```sh
python scripts/review_cycle.py fetch TRAINING_RUN --key .private/dpo.pem
```

This saves the native adapter and report under `reviews/adapter-RUN/`. Next, choose **compare-adapter** in Action 1, enter the **training run number** in **dpo_run**, reuse the **key ID**, and pick a film outside the training rows and earlier probes. Import the result with `python scripts/review_cycle.py fetch COMPARISON_RUN --key .private/dpo.pem`. A/B assignment is made once on import and survives app restarts. Save the artifacts locally before GitHub retention expires.

## Words, tools and feedback

- **Word structures.** The active file is the private [2025 word_structures.py at the pinned commit](https://github.com/patruff/chuck2025/blob/783c52912c0be25ccc27fbca768548c46c977a29/word_structures.py). The [bank loader](pipeline/march_source.py) verifies its SHA-256 and reads the literal `funny_words` and `custom_phones` assignments. For local runs, use an adjacent `chuck2025` checkout or pass `--march-word-bank /path/to/word_structures.py` to the RunPod launcher. Actions use the repository secret `CHUCKLES_2025_BANK_GZ` (base64 of the gzipped pinned file).
- **smolagents.** Both paths load the pinned Hugging Face tools [parody-suggestions](https://huggingface.co/spaces/patruff/parody-suggestions/tree/8147e43f9acfb3af5123fd6fd329c112d8ff7a6b) and [word-phone](https://huggingface.co/spaces/patruff/word-phone/tree/044bccae4f60f213a2246e69c03558a52c6525ff). Suggestions are looked up before drafting. The [RunPod CodeAgent](pipeline/generate_march.py) calls `record_drafts` and executes `word_phone_tool` on its nominees; the [CPU runner](pipeline/blind_film_run.py) calls the same tools between model drafting and selection. Both retain the model’s returned thinking and real observations. [Colab reference](notebooks/Chuckles_RunPod.ipynb)
- **Judging.** One good change can carry the title. A listed word and an invented word compete on the same terms. Check the exact spellings, then let the model choose by audience fit among titles clearing the inclusive March suggestion-score floor of 0.60. The separate word-phone score is displayed alongside it. If revised nominees all miss, the checked fallback is marked below floor.
- **Human guidance.** The kickoff prompt includes selected human judgments and distinguishes a weak winner from a good joke. In **runpod** mode an optional tracked [guidance text file](guidance/human-feedback.txt) lets Qwen propose a creative addendum. The artifact records the applied revision and a `next-feedback.txt` for the next round.
- **Secrets.** RunPod uses `RUNPOD_API_KEY` and tears down each rented pod in the launcher's cleanup path. Training preparation sets `CHUCKLES_DPO_ID` (the compressed request) and `CHUCKLES_DPO_KEY_ID` (recipient key), with ID replaced by the printed suffix. Delete the request secret after training. Keep the recipient key for comparisons and local decryption. The full review export, local comparison bundles and private key do not need to be published.

# Chuckles

Chuckles writes movie-title parodies for an adult audience. GitHub Pages hosts the reviewer, GitHub Actions runs inference and training, and Hugging Face stores the reviews and adapters.

**[Open Review parodies](https://patruff.github.io/chuck26/)** to judge the latest completed inference. The collection menu also includes the 185-film MovieLens catalog and earlier feedback runs.

1. Read both pitches. Expand **Phonetic checks** or **Recorded reasoning** to inspect the original comparison.
2. Choose **Prefer A**, **Prefer B**, or **Neither works**. Add a note if you like and continue through the queue. **Save reviews** saves your pending choices together. Reopen the page and connect to load them again.
3. Choose **Train new adapter** to train on saved A/B choices between distinct titles. GitHub Actions pins the dataset revision, fits the LoRA, reloads it in llama.cpp, and publishes the checked GGUF to the private model repository.
4. Choose **Run inference**, enter movie ranks such as `35`, `1-100`, or `1-5,12,35-40`, or upload a one-column CSV. Choose the base model, the latest published adapter, or **Base model on RunPod GPU**. Completed pairs appear automatically in the review page. A CSV accepts an optional `input` or `title` header. Put one input on each row and quote inputs containing commas.

### Connect your accounts

The page is public. Votes, notes, full traces, and training snapshots are stored in the [private review dataset](https://huggingface.co/datasets/patruff/chuck26-reviews). Under **Connect account**, enter a Hugging Face token with write access to that dataset. A token stays in the current tab’s memory. Draft votes stay in this browser until saved.

For one-click training and inference, also enter a fine-grained GitHub token scoped to **patruff/chuck26** with **Actions: write**. With just the Hugging Face token, **Prepare GitHub run** opens the corresponding GitHub workflow. Choose **Run workflow** and use the displayed revision to run exactly those settings.

Adapters are versioned by GitHub run in the [private adapter repository](https://huggingface.co/patruff/chuck26-adapters). Each manifest identifies the base weights, adapter checksum, and training dataset. The private dataset keeps the DPO rows and comparison report from each hosted training run. [Training details](training/gguf_dpo/README.md)

### Other workflows

- [Generate with an edited audience prompt or RunPod](https://github.com/patruff/chuck26/actions/workflows/human-guidance.yml). The prompt default also applies to hosted inference. Its model and weight revisions are [pinned in the repo](config/model.json).
- In the inference Action, use **rank** for inclusive ranges and comma-separated lists. Overlapping ranks are processed once, in the order entered. **csv_file** accepts a committed CSV path and **csv_revision** accepts the dataset revision from a review-page upload. **preview** produces an `inference-input/selected.csv` artifact before renting a pod. RunPod processes up to 25 inputs per pod, one pod at a time. The Action combines all successful batches into one ordered review collection. If a batch fails, its diagnostics and the completed batch artifacts remain available, and GitHub’s **Re-run failed jobs** retries that work.
- [Generate or resume a RunPod batch](https://github.com/patruff/chuck26/actions/workflows/film-review-batch.yml). Completed pairs are archived for the page.
- RunPod allows up to 12 minutes for the host to start the container and up to 8 minutes from container startup for the model to load, with an 18-minute overall startup limit. The pod downloads the weights in eight parallel ranges and verifies the assembled file against the pinned SHA-256. The Actions log reports the stage and downloaded bytes. For a single run, download `runpod-diagnostics` for the last startup state, elapsed time and teardown confirmation. Bulk runs store the same `run.json` in the shard artifact.
- [Refresh the review page](https://github.com/patruff/chuck26/actions/workflows/review-pages.yml), optionally supplying a specific completed inference run.
- Historical recipient-encrypted comparisons still open with their original key through [the local review tool](scripts/review_cycle.py). The [local Flask reviewer](scripts/review_films.py) also remains available for offline files.

## Words, tools and feedback

- **Word structures.** The active file is the private [2025 word_structures.py at the pinned commit](https://github.com/patruff/chuck2025/blob/783c52912c0be25ccc27fbca768548c46c977a29/word_structures.py). The [bank loader](pipeline/march_source.py) verifies its SHA-256 and reads the literal `funny_words` and `custom_phones` assignments. For local runs, use an adjacent `chuck2025` checkout or pass `--march-word-bank /path/to/word_structures.py` to the RunPod launcher. Actions use the repository secret `CHUCKLES_2025_BANK_GZ` (base64 of the gzipped pinned file).
- **smolagents.** Both paths load the pinned Hugging Face tools [parody-suggestions](https://huggingface.co/spaces/patruff/parody-suggestions/tree/8147e43f9acfb3af5123fd6fd329c112d8ff7a6b) and [word-phone](https://huggingface.co/spaces/patruff/word-phone/tree/044bccae4f60f213a2246e69c03558a52c6525ff). Suggestions are looked up before drafting. The [RunPod CodeAgent](pipeline/generate_march.py) calls `record_drafts` and executes `word_phone_tool` on its nominees; the [CPU runner](pipeline/blind_film_run.py) calls the same tools between model drafting and selection. Both retain the model’s returned thinking and real observations. [Colab reference](notebooks/Chuckles_RunPod.ipynb)
- **Judging.** One good change can carry the title. A listed word and an invented word compete on the same terms. Check the exact spellings, then let the model choose by audience fit among titles clearing the inclusive March suggestion-score floor of 0.60. The separate word-phone score is displayed alongside it. If revised nominees all miss, the checked fallback is marked below floor.
- **Human guidance.** The kickoff prompt includes selected human judgments and distinguishes a weak winner from a good joke. In **runpod** mode an optional tracked [guidance text file](guidance/human-feedback.txt) lets Qwen propose a creative addendum. The artifact records the applied revision and a `next-feedback.txt` for the next round. A push changing `guidance/human-feedback.txt` on main also starts the existing ten-title RunPod feedback run.
- **Secrets.** RunPod uses `RUNPOD_API_KEY` and tears down each rented pod in the launcher's cleanup path. The legacy sealed request flow sets `CHUCKLES_DPO_ID` (the compressed request) and `CHUCKLES_DPO_KEY_ID` (recipient key), with ID replaced by the printed suffix. Delete the request secret after training. Keep the recipient key for comparisons and local decryption. Hosted jobs access the private Hugging Face repositories using the Actions secret `HUGGINGFACE_API_KEY`. Repository variables `CHUCKLES_REVIEW_DATASET` and `CHUCKLES_ADAPTER_REPO` can override the default destinations.

This command provisions the bank secret from your verified checkout.

```sh
gzip -c /path/to/chuck2025/word_structures.py | base64 | tr -d '\n' | gh secret set CHUCKLES_2025_BANK_GZ -R patruff/chuck26
```

The [previous repository layout](https://github.com/patruff/chuck26/tree/541b1f34053467e994447b213f4ee72c4be69364) contains the retired Drive, CSV, SFT and 32B experiments.

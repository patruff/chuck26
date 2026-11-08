# DPO smoke test for the 27B parody model

The [test workflow](../../.github/workflows/dpo-smoke.yml) runs the same RVN Q4_K_M multilingual GGUF used for the film batch. The model revision and SHA-256 are pinned. It runs on a CPU runner and produces a native llama.cpp adapter.

The test reads conversational preference pairs from the review export. It scores the chosen and rejected titles, including the end-of-turn token, after the shared non-thinking Qwen prompt. The frozen model supplies reference logits and the final hidden states. A rank-4 LoRA on `output.weight` learns from the difference between the chosen and rejected sequence log probabilities using the sigmoid DPO objective. The smoke test uses twelve full-batch AdamW steps and a beta of 0.1.

The test saves the adapter, reloads it in a fresh llama.cpp process and checks the chosen-minus-rejected margins against the training calculation. It then generates titles with the unmodified base and the reloaded adapter on identical prompts with greedy decoding. The comparison file records the prompts, per-film margins, generated titles, optimizer settings and checksums.

## Use the adapter

Place the original `RVN-Q4_K_M-multilingual.gguf` and the exported `chuckles-27b-dpo.gguf` on the inference machine. Start the adapter-enabled server with a recent llama.cpp build.

```sh
llama-server -m RVN-Q4_K_M-multilingual.gguf --lora chuckles-27b-dpo.gguf --jinja
```

For the base comparison, start the server with the same model and omit the `--lora` argument. Send the same review-format movie prompt and suggestions to each server with `temperature: 0` and `chat_template_kwargs: {"enable_thinking": false}`. The comparison JSON includes the exact probe messages used in the run.

The GitHub workflow receives the reviewed pairs through a one-run secret. Its downloadable artifact contains the adapter and comparison encrypted for the request's recipient. The review comments remain metadata in the source export.

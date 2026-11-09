# 27B GGUF DPO

[Action 2](../../.github/workflows/dpo-smoke.yml) uses the [same pinned weights](../../config/model.json) as inference. It scores both complete titles after the same non-thinking chat prefix. The frozen model supplies reference logits and final hidden states. A rank-4 LoRA on `output.weight` learns from chosen-minus-rejected sequence log probabilities using sigmoid DPO, beta 0.1 and twelve full-batch AdamW updates.

The trainer saves the GGUF adapter, reloads it in a fresh llama.cpp process and checks that the native margins match the trained policy. The comparison report includes the input and adapter checksums, loss, per-film margins, and greedy generations with and without the adapter. The reviewed file and its comments stay local by default. [Prepare training, fetch the adapter and start the next comparison](../../README.md#train-and-judge-the-next-adapter)

Start llama.cpp with the downloaded adapter and its original base file.

```sh
llama-server -m RVN-Q4_K_M-multilingual.gguf --lora chuckles-27b-dpo.gguf --jinja
```

For a base comparison, start the same server without `--lora`. The DPO probe messages and decoding settings are in the report. The interactive tool-based comparison uses the editable audience instructions from the [inference form](../../.github/workflows/human-guidance.yml).

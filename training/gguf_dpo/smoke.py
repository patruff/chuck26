"""DPO fit on a frozen GGUF's output projection, followed by native adapter inference."""
import argparse
import hashlib
import json
import struct
import subprocess
from pathlib import Path

import gguf
import numpy as np
import torch
import torch.nn.functional as F
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer

MODEL = "0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF"
REVISION = "ab8d7bd81d03117501e4da60cccdff4c80f0eea0"
WEIGHT_SHA = "9ed40ccc8b8432f38b9a85d0ca67928167f5719e8c10bd299e56d34facaf6e61"


def prepare(data, assets, folder):
    tokenizer = Tokenizer.from_file(str(assets / "tokenizer.json"))
    template = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True).from_string(
        (assets / "chat_template.jinja").read_text()
    )
    def encode(messages, generate):
        rendered = template.render(messages=messages, add_generation_prompt=generate,
                                   enable_thinking=False)
        return tokenizer.encode(rendered, add_special_tokens=False).ids

    rows = data["rows"]
    if not rows or len({r["pair_id"] for r in rows}) != len(rows):
        raise ValueError("Expected unique reviewed preferences")
    plans, lengths = [], []
    for r in rows:
        prefix = encode(r["prompt"], True)
        if len(prefix) == 0:
            raise ValueError("Empty prompt")
        line_lengths = []
        for side in ("chosen", "rejected"):
            title = r[side]
            if len(title) != 1 or title[0]["role"] != "assistant" or not title[0]["content"].strip():
                raise ValueError("Missing assistant title")
            # Score the title and its end-of-turn token. Drop trailing template whitespace.
            ids = encode(r["prompt"] + title, False)
            eot = tokenizer.token_to_id("<|im_end|>")
            if ids[:len(prefix)] != prefix or eot not in ids[len(prefix):]:
                raise ValueError("Full chat does not continue the exact inference prefix")
            ids = ids[:ids.index(eot, len(prefix)) + 1]
            line_lengths.append(len(ids) - len(prefix))
            plans.append([len(ids), len(prefix), 0, *ids])
        if r["chosen"] == r["rejected"]:
            raise ValueError("Identical chosen and rejected titles")
        lengths.extend(line_lengths)
    probes = []
    for p in data["probes"]:
        ids = encode(p["prompt"], True)
        probes.append([len(ids), len(ids), 20, *ids])
    for name, records in [("features", plans), ("greedy", probes)]:
        (folder / (name + ".txt")).write_text("".join(" ".join(map(str, a)) + "\n" for a in records))
    return tokenizer, lengths


def run_native(args, mode, adapter, label):
    output = args.work / (label + ".bin")
    subprocess.run([args.driver, args.model, adapter or "-", mode,
                    args.work / (mode + ".txt"), output], check=True)
    return output


def features(path, n):
    with path.open("rb") as f:
        vocab, width = struct.unpack("<ii", f.read(8))
        dtype = np.dtype([("target","<i4"),("hidden","<f4",width),("logits","<f4",vocab)])
        data = np.fromfile(f, dtype=dtype)
    if len(data) != n or not all(np.isfinite(data[k]).all() for k in ("hidden", "logits")):
        raise ValueError("Incomplete or nonfinite native forward pass")
    return tuple(torch.from_numpy(data[k].copy()) for k in ("logits", "hidden", "target"))


def margins(logits, targets, lengths):
    token_scores = F.log_softmax(logits, dim=-1).gather(1, targets.long().unsqueeze(1)).flatten()
    seq = torch.stack([v.sum() for v in torch.split(token_scores, lengths)])
    return seq[::2] - seq[1::2]


def save_adapter(path, model, a, b):
    reader = gguf.GGUFReader(model)
    arch = reader.get_field("general.architecture").contents()
    if not any(t.name == "output.weight" for t in reader.tensors):
        raise ValueError("Model has no independent output projection")
    writer = gguf.GGUFWriter(path, arch)
    writer.add_type(gguf.GGUFType.ADAPTER)
    writer.add_string(gguf.Keys.Adapter.TYPE, "lora")
    writer.add_float32(gguf.Keys.Adapter.LORA_ALPHA, a.shape[0])
    writer.add_tensor("output.weight.lora_a", a.detach().numpy().astype(np.float16))
    writer.add_tensor("output.weight.lora_b", b.detach().numpy().astype(np.float16))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def greedy_text(path, tokenizer, probes):
    gathered = []
    with path.open("rb") as f:
        vocab, _ = struct.unpack("<ii", f.read(8))
        for _ in probes:
            (n,) = struct.unpack("<i", f.read(4))
            if not 0 <= n <= 20:
                raise ValueError("Invalid generation length")
            ids = list(struct.unpack(f"<{n}i", f.read(n * 4)))
            if any(i < 0 or i >= vocab for i in ids):
                raise ValueError("Invalid generated token")
            gathered.append({"text":tokenizer.decode(ids, skip_special_tokens=False),
                             "tokens":n, "reached_limit":n == 20})
        if f.read(1):
            raise ValueError("Unexpected generation output")
    return gathered


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("input","model","assets","driver","work","output"):
        p.add_argument("--"+name, type=Path, required=True)
    args = p.parse_args()
    args.work.mkdir(exist_ok=True, parents=True)
    args.output.mkdir(exist_ok=True, parents=True)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    data = json.loads(args.input.read_text())
    tokenizer, lengths = prepare(data, args.assets, args.work)
    base, hidden, targets = features(run_native(args, "features", None, "base-features"), sum(lengths))
    ref = margins(base, targets, lengths).detach()
    rank, beta, lr, steps = 4, 0.1, 0.001, 12
    a = torch.nn.Parameter(torch.randn(rank, hidden.shape[1]) / hidden.shape[1]**0.5)
    b = torch.nn.Parameter(torch.zeros(base.shape[1], rank))
    optimizer = torch.optim.AdamW([a,b], lr=lr, weight_decay=0)
    loss_trace = []
    for _ in range(steps):
        policy = base + (hidden @ a.T) @ b.T
        loss = -F.logsigmoid(beta * (margins(policy, targets, lengths) - ref)).mean()
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite DPO loss")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_trace.append(loss.item())
    adapter = args.output / "chuckles-27b-dpo.gguf"
    save_adapter(adapter, args.model, a, b)
    print("Saved native GGUF output-projection adapter. Checking it in a fresh llama.cpp process.", flush=True)
    with torch.no_grad():
        expected = margins(base + (hidden @ a.detach().half().float().T) @ b.detach().half().float().T,
                           targets, lengths)
    trained, _, new_targets = features(run_native(args,"features",adapter,"reloaded-features"),sum(lengths))
    if not torch.equal(targets, new_targets):
        raise ValueError("Target tokens changed")
    actual = margins(trained, targets, lengths)
    delta = actual - ref
    error = (actual - expected).abs().max().item()
    final_loss = -F.logsigmoid(beta * delta).mean().item()
    if error > 0.15 or final_loss >= loss_trace[0] or not torch.isfinite(actual).all():
        raise ValueError(f"Native reload did not reproduce the trained policy: error {error}, loss {final_loss}")
    print(f"DPO loss {loss_trace[0]:.4f} -> {final_loss:.4f}; native reload verified", flush=True)
    baseline = greedy_text(run_native(args,"greedy",None,"base-text"),tokenizer,data["probes"])
    adapted = greedy_text(run_native(args,"greedy",adapter,"adapter-text"),tokenizer,data["probes"])
    report = dict(model=MODEL,revision=REVISION,base_weight_sha256=WEIGHT_SHA,
        adapter_sha256=hashlib.sha256(adapter.read_bytes()).hexdigest(), dataset_sha256=data["dataset_sha256"],
        settings=dict(target="output.weight",rank=rank,alpha=rank,beta=beta,learning_rate=lr,optimizer="AdamW",
                      full_batch_steps=steps,seed=42,greedy_max_new_tokens=20,thinking=False),
        initial_loss=loss_trace[0],final_reloaded_loss=final_loss, training_loss=loss_trace,
        native_margin_max_abs_error=error,
        preferences=[dict(film=row["film"],chosen=row["chosen"][0]["content"],rejected=row["rejected"][0]["content"],
                    base_margin=ref[i].item(),adapted_margin=actual[i].item(),change=delta[i].item())
                    for i,row in enumerate(data["rows"])],
        generations=[dict(film=p["film"],reviewed=p["reviewed"],prompt=p["prompt"],base=v,adapter=w)
                     for p,v,w in zip(data["probes"],baseline,adapted)])
    (args.output/"comparison.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")


if __name__ == "__main__":
    main()

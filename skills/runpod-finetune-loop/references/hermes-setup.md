# Wiring this skill into Hermes Agent

Hermes (Nous Research) loads skills from `~/.hermes/skills/` in the agentskills.io
SKILL.md format — which is exactly what this folder is — and configures MCP servers
in `~/.hermes/config.yaml` under `mcp_servers:`. Steps below get the whole loop
running.

## 1. Install the skill

Drop the folder into Hermes's skills directory, then have Hermes rescan:

```bash
cp -r runpod-finetune-loop ~/.hermes/skills/
```

Inside a Hermes session:

```
/reload-skills          # re-scans ~/.hermes/skills/ for the new skill
/skills                 # confirm runpod-finetune-loop is listed
/skill runpod-finetune-loop   # load it into the current session
```

Hermes also auto-discovers skills by description, so once it's reloaded the agent
will pull it in when a task matches the trigger (fine-tune on RunPod, push/pull an
adapter, etc.). The `/skill` command just forces it into the active session.

## 2. Give Hermes the MCPs it needs

Add these under `mcp_servers:` in `~/.hermes/config.yaml`. The `tools.include`
filter keeps the surface small (Hermes's guidance is "connect the right thing, with
the smallest useful surface"), which also saves context tokens.

```yaml
mcp_servers:
  # --- RunPod: pod lifecycle (REQUIRED) ---
  # Official RunPod MCP server (github.com/runpod/runpod-mcp), Node-based, run via
  # npx. Current package: @runpod/mcp-server.
  runpod:
    command: "npx"
    args: ["-y", "@runpod/mcp-server@latest"]
    env:
      RUNPOD_API_KEY: "${RUNPOD_API_KEY}"
    tools:
      include:
        - create-pod
        - get-pod
        - list-pods
        - stop-pod
        - delete-pod
        - create-endpoint   # optional: only if you later serve the merged model

  # --- Hugging Face: verify the pushed adapter + model/dataset search (RECOMMENDED) ---
  # Remote HTTP server at https://huggingface.co/mcp, authenticated with HF_TOKEN.
  huggingface:
    url: "https://huggingface.co/mcp"
    headers:
      Authorization: "Bearer ${HF_TOKEN}"

  # --- GitHub: log run results to the team's issues (OPTIONAL) ---
  # Pat already runs the Evarian team async via GitHub Issues; handy for recording
  # eval scores. Use a read/write-scoped PAT and keep the tool surface minimal.
  github:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "${GITHUB_PERSONAL_ACCESS_TOKEN}"
    tools:
      include: [create_issue, update_issue, list_issues]
```

Then, in a Hermes session:

```
/reload-mcp             # picks up the new servers without a restart
```

Ask Hermes "which MCP-backed tools are available right now?" to confirm the RunPod
and Hugging Face tools registered. If a server needs OAuth, Hermes prints an
authorize URL; on a headless/remote host use its paste-back flow or an SSH port
forward (see Hermes MCP docs).

Alternatively, add a server from the CLI instead of editing YAML by hand:

```bash
hermes mcp add runpod --command npx --args "-y @runpod/mcp-server@latest"
```

## 3. Set the secrets

Put these in `~/.hermes/.env` (or the profile's env) so both the MCPs and the
training scripts can read them:

```
RUNPOD_API_KEY=...
HF_TOKEN=...
GITHUB_PERSONAL_ACCESS_TOKEN=...   # only if using the GitHub MCP
```

`HF_TOKEN` must also be passed into the pod's container env (the skill does this) so
`train_lora.py` can `push_to_hub`. Run `/reload` in-session after editing `.env`.

## 4. (Optional) Run it unattended on a cron

This is the "improve while I'm away" pattern. Hermes's built-in cron can fire a
natural-language task on a schedule and deliver the result to any platform:

```
/cron add "every Sunday 2am: run the runpod-finetune-loop skill to fine-tune
<base-model> on /workspace/data/latest.jsonl, push the adapter to
<user>/<model>-adapter, terminate the pod, run the eval prompts, and DM me the
pass rate" --deliver telegram
```

Because the skill terminates the pod in step 5 and only the tiny adapter persists,
an unattended run can't leave an expensive GPU billing overnight — provided the
RunPod `delete-pod` tool is in the include-list above so Hermes can actually call
it.

## Division of labour (what drives what)

- **RunPod MCP** → step 1 provision, step 5 terminate.
- **Hugging Face MCP** → step 4 verify the push; also handy for picking a base model.
- **Hermes native terminal/SSH** → steps 2, 3, 6 (stage files, run `train_lora.py`
  on the pod, run `test_inference.py` afterward).
- **`scripts/orchestrate.py`** → fallback for pod lifecycle if the RunPod MCP is down.
- **GitHub MCP (optional)** → step 7 logging.

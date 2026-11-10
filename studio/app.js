import { ReviewHub } from "./hub.js";

const $ = (id) => document.getElementById(id);
const node = (tag, text, cls) => {
  const n = document.createElement(tag);
  if (text !== undefined) n.textContent = text;
  if (cls) n.className = cls;
  return n;
};
const clicker = (text, fn, cls) => {
  const b = node("button", text, cls);
  b.type = "button";
  b.onclick = fn;
  return b;
};
const option = (value, text) => new Option(text, value);
const message = (text) => {
  $("status").textContent = text;
};
const error = (e) => message(e.message || String(e));
const readMap = (key) => {
  try {
    return new Map(JSON.parse(localStorage.getItem(key) || "[]"));
  } catch {
    return new Map();
  }
};
let manifest,
  collection,
  pairs = [],
  index = 0,
  loading = 0,
  hub,
  gitToken = "",
  saved = new Map(),
  savedRevision,
  drafts = new Map(),
  saving = false,
  draftKey;
const traceCache = new Map();
let conflictId = "";
const current = () => pairs[index];
const selection = (id) => drafts.get(id) || saved.get(id);
function keepDrafts() {
  try {
    localStorage.setItem(draftKey, JSON.stringify([...drafts]));
  } catch {
    message(
      "The browser could not keep a local draft. Save before leaving this tab.",
    );
  }
}
function stateLine() {
  const p = current();
  if (!p) return;
  const count = drafts.size,
    votes = [...saved.values()].filter(
      (v) => v.choice === "a" || v.choice === "b",
    ).length;
  $("state").textContent = count
    ? `${count} unsaved review${count === 1 ? "" : "s"}`
    : hub
      ? `${saved.size} saved · ${votes} A/B choices`
      : "Votes stay in this browser until you save.";
  $("comment").disabled = saving;
  for (const b of document.querySelectorAll(".pick, #alternatives button"))
    b.disabled = saving;
  $("save").disabled = !count || saving;
  $("save").textContent = saving ? "Saving…" : "Save reviews";
}
function setEdit(choice, feedback) {
  const p = current();
  if (!p || saving) return;
  const base = saved.get(p.id),
    prior = drafts.get(p.id);
  if (choice === base?.choice && feedback === (base?.feedback || ""))
    drafts.delete(p.id);
  else if ((choice === "clear" || !choice) && !base && !feedback)
    drafts.delete(p.id);
  else
    drafts.set(p.id, {
      pair_id: p.id,
      collection: collection.id,
      film: p.film,
      blind: p.blind,
      titles: p.sides.map((s) => s.title),
      choice: choice || "",
      feedback,
      expected: prior?.expected ?? base?.updated ?? "",
    });
  keepDrafts();
  stateLine();
  queue();
}
function queue() {
  const listing = $("films"),
    search = $("search").value.trim().toLowerCase();
  listing.replaceChildren();
  pairs.forEach((p, i) => {
    if (!p.film.title.toLowerCase().includes(search)) return;
    const b = clicker(p.film.title, () => go(i));
    if (i === index) {
      b.className = "active-film";
      b.setAttribute("aria-current", "true");
    }
    if (drafts.has(p.id)) b.append(node("small", "Draft"));
    else if (saved.has(p.id)) b.append(node("small", "Saved"));
    listing.append(b);
  });
  if (!listing.childElementCount)
    listing.append(node("p", "No films match.", "hint"));
  $("total").textContent = "· " + pairs.length;
}
async function loadCollection(id) {
  const load = ++loading;
  const selected = manifest.collections.find((c) => c.id === id);
  if (!selected) return;
  $("comparison").hidden = true;
  $("empty").hidden = false;
  $("empty").textContent = "Loading comparisons…";
  $("films").replaceChildren();
  $("collection").value = id;
  const r = await fetch("data/" + encodeURIComponent(id) + ".json");
  if (!r.ok)
    throw Error("Could not load this collection. Refresh to try again.");
  const list = await r.json();
  if (load !== loading) return;
  pairs = list;
  collection = selected;
  $("search").value = "";
  if (!pairs.length) {
    $("empty").textContent = "This collection has no complete comparisons.";
    return;
  }
  go(
    Math.max(
      0,
      pairs.findIndex((p) => !saved.has(p.id) && !drafts.has(p.id)),
    ),
  );
}
function go(i) {
  index = Math.min(pairs.length - 1, Math.max(0, i));
  render();
}
function fold(text) {
  const d = node("details");
  d.append(node("summary", text));
  return d;
}
async function traces() {
  if (!hub) throw Error("Connect Hugging Face to read the original trace.");
  const key = manifest.revision + "/" + collection.id;
  if (!traceCache.has(key)) {
    traceCache.set(
      key,
      hub.traces(collection.id, manifest.revision).catch((e) => {
        traceCache.delete(key);
        throw e;
      }),
    );
  }
  return traceCache.get(key);
}
function render() {
  const p = current();
  if (!p) return;
  $("comparison").hidden = false;
  $("empty").hidden = true;
  $("position").textContent = `Film ${index + 1} of ${pairs.length}`;
  $("film").textContent = p.film.title;
  $("prev").disabled = index === 0;
  $("next").disabled = index === pairs.length - 1;
  const vote = selection(p.id),
    pitches = $("pitches");
  pitches.replaceChildren();
  p.sides.forEach((s, i) => {
    const letter = i ? "B" : "A",
      choice = letter.toLowerCase(),
      selected = vote?.choice === choice;
    const card = node(
      "article",
      undefined,
      "pitch" + (selected ? " selected" : ""),
    );
    card.append(node("p", "Pitch " + letter, "pitch-tag"), node("h3", s.title));
    if (s.why) card.append(node("p", s.why, "why"));
    const evidence = node("div", undefined, "evidence"),
      checks = fold("Phonetic checks");
    const table = node("table"),
      header = node("tr");
    for (const label of ["Change", "March", "Word-phone"])
      header.append(node("th", label));
    table.append(header);
    for (const pair of s.pairs) {
      const tr = node("tr");
      const score = (v) => (typeof v === "number" ? v.toFixed(2) : "—");
      tr.append(
        node("td", pair.compare_to + " → " + pair.word),
        node("td", score(pair.suggestion_score ?? pair.score)),
        node("td", score(pair.word_phone_score)),
      );
      table.append(tr);
    }
    checks.append(table);
    const reasoning = fold("Recorded reasoning");
    reasoning.ontoggle = async () => {
      if (!reasoning.open || reasoning.children.length > 1) return;
      const box = node("div", "Loading…", "hint");
      reasoning.append(box);
      if (!hub) {
        box.replaceChildren(clicker("Connect to read trace", () => connect()));
        return;
      }
      try {
        const records = await traces();
        box.replaceChildren(
          node("pre", records.get(p.id)?.sides[i] || "No trace recorded."),
        );
      } catch (e) {
        box.textContent = e.message;
      }
    };
    evidence.append(checks, reasoning);
    const b = clicker(
      "Prefer " + letter,
      () => {
        setEdit(choice, $("comment").value);
        render();
      },
      "pick",
    );
    b.setAttribute("aria-pressed", String(selected));
    card.append(evidence, b);
    pitches.append(card);
  });
  $("alternatives").replaceChildren();
  for (const [v, label] of [
    ...(p.blind ? [["tie", "Equally funny"]] : []),
    ["neither", "Neither works"],
  ]) {
    const b = clicker(label, () => {
      setEdit(v, $("comment").value);
      render();
    });
    b.setAttribute("aria-pressed", String(vote?.choice === v));
    $("alternatives").append(b);
  }
  if (vote && vote.choice !== "clear")
    $("alternatives").append(
      clicker("Clear vote", () => {
        setEdit("clear", "");
        render();
      }),
    );
  $("comment").value = vote?.feedback || "";
  $("reveal").replaceChildren();
  if (p.blind && saved.has(p.id) && !drafts.has(p.id))
    $("reveal").append(
      clicker("Reveal models", async () => {
        try {
          const r = (await traces()).get(p.id);
          $("reveal").textContent = r.attribution
            .map((a, i) => (i ? "B" : "A") + " · " + a)
            .join(" / ");
        } catch (e) {
          error(e);
        }
      }),
    );
  $("conflict").replaceChildren();
  if (conflictId === p.id && drafts.has(p.id)) {
    $("conflict").append(
      node(
        "p",
        "Saved version: " +
          (saved.get(p.id)?.choice || "cleared") +
          " · " +
          (saved.get(p.id)?.feedback || "No note"),
      ),
      clicker("Use saved version", () => {
        drafts.delete(p.id);
        conflictId = "";
        keepDrafts();
        render();
      }),
      clicker("Keep my edit", () => {
        drafts.get(p.id).expected = saved.get(p.id)?.updated || "";
        conflictId = "";
        keepDrafts();
        render();
      }),
    );
  }
  stateLine();
  queue();
}
function modal(title) {
  const d = $("dialog");
  $("dialog-title").textContent = title;
  const body = $("dialog-body");
  body.replaceChildren();
  if (!d.open) d.showModal();
  return body;
}
function field(container, label, input) {
  const wrap = node("div", undefined, "field");
  const id = "field-" + container.querySelectorAll("label").length;
  input.id = id;
  const l = node("label", label);
  l.htmlFor = id;
  wrap.append(l, input);
  container.append(wrap);
  return input;
}
function link(label, url) {
  const a = node("a", label);
  a.href = url;
  a.target = "_blank";
  a.rel = "noopener noreferrer";
  return a;
}
function connect(after) {
  const body = modal("Connect account");
  const hf = field(body, "Hugging Face write token", node("input"));
  hf.type = "password";
  hf.autocomplete = "off";
  const h = node("p", undefined, "hint");
  h.append(
    "For saving reviews to " + manifest.dataset + ". ",
    link("Create a scoped token", "https://huggingface.co/settings/tokens"),
  );
  body.append(h);
  const gh = field(body, "GitHub Actions token (optional)", node("input"));
  gh.type = "password";
  gh.autocomplete = "off";
  const hint = node("p", undefined, "hint");
  hint.append(
    "Use Actions write access to launch jobs here, or use the GitHub workflow form. ",
    link(
      "Create a scoped token",
      "https://github.com/settings/personal-access-tokens/new?name=Chuckles+review&actions=write",
    ),
  );
  body.append(
    hint,
    node("p", "Tokens are kept only in this tab’s memory.", "hint"),
  );
  const note = node("p", "", "hint"),
    actions = node("div", undefined, "actions");
  const apply = clicker(
    "Connect",
    async () => {
      apply.disabled = true;
      try {
        const client = hf.value.trim()
          ? new ReviewHub(manifest.dataset, hf.value)
          : hub;
        if (!client)
          throw Error(
            "Enter a Hugging Face token for the private review dataset.",
          );
        const result = await client.reviews();
        hub = client;
        saved = result.records;
        savedRevision = result.sha;
        gitToken = gh.value.trim() || gitToken;
        hf.value = "";
        gh.value = "";
        $("connect").textContent = "Account settings";
        $("dialog").close();
        body.replaceChildren();
        message("Connected to the review dataset.");
        render();
        if (after) await after();
      } catch (e) {
        note.textContent = e.message;
      } finally {
        apply.disabled = false;
      }
    },
    "primary",
  );
  actions.append(apply);
  if (hub)
    actions.append(
      clicker("Disconnect", () => {
        hub = null;
        gitToken = "";
        saved = new Map();
        traceCache.clear();
        body.replaceChildren();
        $("dialog").close();
        $("connect").textContent = "Connect account";
        render();
        message("Disconnected.");
      }),
    );
  body.append(note, actions);
}
async function save() {
  if (!hub) {
    connect(save);
    return;
  }
  if (saving || !drafts.size) return;
  const submitting = new Map(drafts);
  saving = true;
  stateLine();
  try {
    const res = await hub.save(submitting);
    saved = res.records;
    savedRevision = res.sha;
    for (const [id, edit] of submitting)
      if (drafts.get(id) === edit) drafts.delete(id);
    keepDrafts();
    conflictId = "";
    message(
      `Saved ${submitting.size} review${submitting.size === 1 ? "" : "s"} to Hugging Face.`,
    );
  } catch (e) {
    if (e.conflict) {
      saved = e.records;
      savedRevision = e.sha;
      conflictId = e.conflict;
      const edit = drafts.get(e.conflict);
      if (collection.id !== edit.collection)
        await loadCollection(edit.collection);
      const n = pairs.findIndex((p) => p.id === e.conflict);
      if (n >= 0) go(n);
    }
    error(e);
  } finally {
    saving = false;
    render();
  }
}
function workflowUrl(file) {
  return (
    "https://github.com/" + manifest.repository + "/actions/workflows/" + file
  );
}
async function dispatch(file, inputs) {
  const r = await fetch(
    "https://api.github.com/repos/" +
      manifest.repository +
      "/actions/workflows/" +
      file +
      "/dispatches",
    {
      method: "POST",
      headers: {
        Authorization: "Bearer " + gitToken,
        Accept: "application/vnd.github+json",
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ ref: "main", inputs }),
    },
  );
  if (!r.ok)
    throw Error(
      "GitHub did not start the workflow (" +
        r.status +
        "). Check Actions write access or use the GitHub form.",
    );
  $("dialog").close();
  message("Workflow submitted. ");
  $("status").append(link("View run on GitHub", workflowUrl(file)));
}
function validateCsv(text) {
  const titles = text
    .replace(/^\uFEFF/, "")
    .split(/\r\n|\n|\r/)
    .filter((line) => line.trim())
    .map((line) => {
      const quoted = /^"((?:[^"]|"")*)"$/.exec(line);
      if (quoted) return quoted[1].replaceAll('""', '"').trim();
      if (/[",]/.test(line))
        throw Error(
          "Use one title per row and quote titles containing commas.",
        );
      return line.trim();
    });
  if (
    [
      "title",
      "movie",
      "input",
      "parody",
      "input parody",
      "input parodies",
      "input_title",
    ].includes(titles[0]?.toLowerCase())
  )
    titles.shift();
  if (!titles.length || titles.some((title) => !title))
    throw Error("The CSV needs at least one title and one column.");
}
function jobs(kind) {
  if (!hub) {
    connect(() => jobs(kind));
    return;
  }
  const train = kind === "train",
    body = modal(train ? "Train new adapter" : "Run inference");
  const file = train ? "dpo-smoke.yml" : "studio-inference.yml";
  const count = [...saved.values()].filter(
    (v) => ["a", "b"].includes(v.choice) && v.titles?.[0] !== v.titles?.[1],
  ).length;
  if (train && (drafts.size || !count)) {
    body.append(
      node(
        "p",
        drafts.size
          ? "Save your pending reviews before training."
          : "Save at least one A/B preference between distinct titles before training.",
      ),
    );
    return;
  }
  let movie, model, upload;
  if (train) {
    body.append(
      node(
        "p",
        `Train from ${count} saved A/B preference${count === 1 ? "" : "s"}. The checked GGUF adapter will be published to ${manifest.models}.`,
      ),
    );
  } else {
    movie = field(body, "Catalog movie", node("select"));
    for (const m of manifest.movies)
      movie.add(option(m.rank, m.rank + " · " + m.title));
    movie.value = "35";
    model = field(body, "Model", node("select"));
    model.add(option("base", "Base model"));
    model.add(option("latest-adapter", "Latest trained adapter"));
    upload = field(body, "Upload CSV instead (optional)", node("input"));
    upload.type = "file";
    upload.accept = ".csv,text/csv";
    upload.onchange = () => {
      movie.disabled = Boolean(upload.files.length);
    };
    body.append(
      node(
        "p",
        "Use one title per row. Quote titles containing commas. Input titles and the generated pitches appear on the public page.",
        "hint",
      ),
    );
  }
  const note = node("p", "", "hint");
  const button = clicker(
    gitToken
      ? train
        ? "Start training"
        : "Start inference"
      : "Prepare GitHub run",
    async () => {
      button.disabled = true;
      try {
        let revision = train ? savedRevision : await hub.head(),
          inputs;
        if (train) inputs = { revision };
        else {
          const csv = upload.files[0];
          const req = {
            source: csv ? "csv" : "catalog",
            rank: movie.value,
            model: model.value,
          };
          const files = { "requests/inference.json": JSON.stringify(req) };
          if (csv) {
            const text = await csv.text();
            validateCsv(text);
            files["inputs/latest.csv"] = text;
          }
          revision = await hub.commit(
            files,
            revision,
            "Prepare inference inputs",
          );
          inputs = { ...req, revision };
        }
        if (gitToken) {
          await dispatch(file, inputs);
          return;
        }
        note.replaceChildren(
          node(
            "span",
            train
              ? "Open GitHub and choose Run workflow to train from the saved reviews."
              : "Open GitHub and choose Run workflow with “page” as the source to use these settings.",
          ),
          node("br"),
          link("Open GitHub workflow ↗", workflowUrl(file)),
          node("br"),
          node(
            "span",
            "To pin this exact dataset snapshot, use revision " +
              revision +
              ".",
          ),
        );
      } catch (e) {
        note.textContent = e.message;
      } finally {
        button.disabled = false;
      }
    },
    "primary",
  );
  body.append(note, button);
}
$("connect").onclick = () => connect();
$("close").onclick = () => $("dialog").close();
$("dialog").addEventListener("close", () => {
  if (!$("dialog").open) $("dialog-body").replaceChildren();
});
$("infer").onclick = () => jobs("infer");
$("train").onclick = () => jobs("train");
$("save").onclick = save;
$("next").onclick = () => go(index + 1);
$("prev").onclick = () => go(index - 1);
$("search").oninput = queue;
$("comment").oninput = () =>
  setEdit(selection(current().id)?.choice, $("comment").value);
$("collection").onchange = () =>
  loadCollection($("collection").value).catch(error);
if (matchMedia("(max-width:780px)").matches) $("queue").open = false;
async function init() {
  const r = await fetch("data/index.json", { cache: "no-store" });
  if (!r.ok) throw Error("Catalog unavailable. Reload to retry.");
  manifest = await r.json();
  draftKey = "chuckles-drafts:" + manifest.dataset;
  drafts = readMap(draftKey);
  for (const id of ["connect", "train", "infer"]) $(id).disabled = false;
  for (const c of manifest.collections)
    $("collection").add(option(c.id, c.label + " (" + c.count + ")"));
  await loadCollection(manifest.collections[0].id);
}
init().catch((e) => {
  $("empty").textContent = e.message;
});

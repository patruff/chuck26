// Credentials live in this tab's memory. All writes use the private dataset's current parent.
export class ReviewHub {
  constructor(repo, token, branch = "main") {
    if (!/^[\w.-]+\/[\w.-]+$/.test(repo))
      throw Error("Invalid dataset repository.");
    this.branch = encodeURIComponent(branch);
    this.token = token.trim();
    this.api = "https://huggingface.co/api/datasets/" + repo;
    this.files = "https://huggingface.co/datasets/" + repo;
  }
  async request(url, options = {}) {
    const response = await fetch(url, {
      ...options,
      cache: "no-store",
      referrerPolicy: "no-referrer",
      headers: { Authorization: "Bearer " + this.token, ...options.headers },
    });
    if (!response.ok) {
      const error = new Error(
        "Hugging Face returned " +
          response.status +
          (response.status === 401 || response.status === 403
            ? ". Check the token’s dataset access."
            : "."),
      );
      error.status = response.status;
      throw error;
    }
    return response;
  }
  async head() {
    const info = await (
      await this.request(this.api + "/revision/" + this.branch)
    ).json();
    if (!/^[a-f\d]{40}$/.test(info.sha))
      throw Error("The dataset revision is unavailable.");
    return info.sha;
  }
  async file(path, sha) {
    return this.request(this.files + "/resolve/" + sha + "/" + path);
  }
  async reviews() {
    const sha = await this.head();
    const contents = await (
      await this.file("reviews/film_preferences.jsonl", sha)
    ).text();
    const records = new Map();
    for (const line of contents.trim().split("\n").filter(Boolean)) {
      const row = JSON.parse(line);
      if (!row.pair_id || records.has(row.pair_id))
        throw Error("Invalid or duplicate saved review.");
      records.set(row.pair_id, row);
    }
    return { sha, records };
  }
  async traces(collection, revision) {
    const file = await this.file(
      "collections/" + encodeURIComponent(collection) + "/traces.jsonl.gz",
      revision,
    );
    const stream = file.body.pipeThrough(new DecompressionStream("gzip"));
    const text = await new Response(stream).text();
    return new Map(
      text
        .trim()
        .split("\n")
        .filter(Boolean)
        .map((line) => {
          const record = JSON.parse(line);
          return [record.id, record];
        }),
    );
  }
  async commit(files, parent, summary) {
    const lines = [{ key: "header", value: { summary, parentCommit: parent } }];
    for (const [path, text] of Object.entries(files)) {
      const bytes = new TextEncoder().encode(text);
      let binary = "";
      for (let i = 0; i < bytes.length; i += 8192)
        binary += String.fromCharCode(...bytes.subarray(i, i + 8192));
      lines.push({
        key: "file",
        value: { path, content: btoa(binary), encoding: "base64" },
      });
    }
    const response = await this.request(this.api + "/commit/" + this.branch, {
      method: "POST",
      headers: { "Content-Type": "application/x-ndjson" },
      body: lines.map((x) => JSON.stringify(x)).join("\n") + "\n",
    });
    return (await response.json()).commitOid;
  }
  async save(drafts) {
    for (let attempt = 0; attempt < 3; attempt++) {
      const { sha, records: saved } = await this.reviews();
      const records = new Map(saved);
      let changed = false;
      for (const [id, edit] of drafts) {
        const old = records.get(id);
        if ((old?.updated || "") !== edit.expected) {
          // A retry after a lost HTTP reply is already complete.
          if (
            (!old && edit.choice === "clear") ||
            (old?.choice === edit.choice && old.feedback === edit.feedback)
          )
            continue;
          const error = new Error(
            "The saved vote for " +
              edit.film.title +
              " changed. Review both versions before saving.",
          );
          Object.assign(error, { conflict: id, records: saved, sha });
          throw error;
        }
        if (edit.choice === "clear") {
          changed = records.delete(id) || changed;
          continue;
        }
        if (
          !["a", "b", "neither", ...(edit.blind ? ["tie"] : [])].includes(
            edit.choice,
          )
        ) {
          throw Error(
            "Choose a vote for " + edit.film.title + " before saving.",
          );
        }
        changed = true;
        const { expected, ...row } = edit;
        records.set(id, {
          ...row,
          pair_id: id,
          updated: new Date().toISOString(),
          ...(row.blind
            ? { first_choice: old?.first_choice || row.choice }
            : {}),
        });
      }
      if (!changed) return { records, sha };
      const ledger = [...records.values()]
        .map((row) => JSON.stringify(row))
        .join("\n");
      try {
        const revision = await this.commit(
          { "reviews/film_preferences.jsonl": ledger ? ledger + "\n" : "" },
          sha,
          "Save parody reviews",
        );
        return { records, sha: revision };
      } catch (e) {
        if (attempt === 2 || (await this.head()) === sha) throw e;
      }
    }
  }
}

"""Use smolagents CodeAgent to create and check movie-title parodies on a GGUF server."""
import argparse
import ast
from difflib import SequenceMatcher
import hashlib
import json
import os
from pathlib import Path
import re

from common import MIN_PHONETIC_SCORE, align_swaps, clean_word, compact_suggestions
from generate_with_reasoning import read_titles
from chuckles_prime.tools import load_parody_tools
from smolagents import ActionStep, CodeAgent, OpenAIModel, Tool, load_tool
from smolagents.monitoring import LogLevel
from humor_batch import RecordDrafts, clears_phonetic_floor, suggested_pairs, title_changes
from human_guidance import BASE_INSTRUCTIONS, revise_prompt, write_next_feedback
from trace_review import write_transcripts
from march_source import BANK_SHA256, BANK_URL, NOTEBOOK

# These Hub sources match the tool classes printed in the linked Colab's saved output.
COLAB_TOOLS = {"patruff/parody-suggestions": "39102988c0309dd7cb96d196f671c0e5ef9c70aa",
               "patruff/word-phone": "b1c920767796dde68f208d4182c7d889be02ee5b"}
RUNNING_MAN_TOOLS = {"patruff/parody-suggestions": "8147e43f9acfb3af5123fd6fd329c112d8ff7a6b",
                     "patruff/word-phone": "044bccae4f60f213a2246e69c03558a52c6525ff"}
RUNNING_MAN_CELL = "tbAjK5rX4tjo"
HUMOR_BATCH_CELL = "GvGq8GcoGbCD"


class ReasoningGGUFModel(OpenAIModel):
    @property
    def supports_stop_parameter(self):
        # The server matches stop strings inside thinking as well as visible text.
        # OpenAIModel applies CodeAgent's stops to the visible content locally instead.
        return False


def running_man_prompt(notebook, title, suggestions, cell_id=RUNNING_MAN_CELL):
    """Render the saved cell's two substitutions without evaluating notebook code."""
    cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    cell = next((c for c in cells if c.get("metadata", {}).get("id") == cell_id), None)
    if cell is None:
        raise ValueError(f"Expected Running Man demonstration cell {cell_id} in {notebook.name}")
    module = ast.parse("".join(cell["source"]))
    fn = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == "generate_parody")
    template = next(n.value for n in ast.walk(fn) if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "prompt" for t in n.targets))
    values = {"title": title, "json.dumps(suggestions, indent=2)": json.dumps(suggestions, indent=2)}
    return "".join(n.value if isinstance(n, ast.Constant) else values[ast.unparse(n.value)] for n in template.values)


def load_word_bank(path):
    """Read literal data from the historical module without importing its code."""
    wanted = {"funny_words", "custom_phones"}
    data = {}
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in wanted:
                data[target.id] = ast.literal_eval(node.value)
    if not isinstance(data.get("funny_words"), list) or not isinstance(data.get("custom_phones"), dict):
        raise ValueError("Expected funny_words and custom_phones in the historical word bank")
    return data["funny_words"], data["custom_phones"]


def suggestions_for(title, suggester, words, phones, cutoff=MIN_PHONETIC_SCORE):
    raw = {}
    for token in title.split():
        word = clean_word(token)
        if word in raw or not word:
            continue
        raw[word] = json.loads(suggester.forward(
            target=word, word_list_str=words, min_similarity=str(cutoff),
            **({"custom_phones": phones} if phones else {}),
        ))
    return raw, {w: r["error"] for w, r in raw.items() if "error" in r}


def notebook_words(path):
    """Read only the literal funny_words list. Notebook code is never executed."""
    candidates = []
    for cell in json.loads(path.read_text(encoding="utf-8"))["cells"]:
        code = "".join(cell.get("source", []))
        if cell["cell_type"] != "code" or not re.search(r"(?m)^funny_words\s*=", code):
            continue
        for stmt in ast.parse(code).body:
            if isinstance(stmt, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "funny_words" for t in stmt.targets):
                words = ast.literal_eval(stmt.value)
                if not isinstance(words, list) or not all(isinstance(w, str) for w in words):
                    raise ValueError("Expected a literal list of words")
                candidates.append(words)
    if len(candidates) != 1 or not candidates[0]:
        raise ValueError("Expected a single nonempty funny_words list in the notebook")
    return candidates[0]


class WordPhoneTool(Tool):
    name = "word_phone_tool"
    description = ("Check an original word against its proposed replacement. Returns a Python dict with "
                   "phones and score; access result['score']. An unknown pronunciation has score None.")
    inputs = {"word": {"type": "string", "description": "Original word"},
              "compare_to": {"type": "string", "description": "Proposed replacement word"}}
    output_type = "object"

    def __init__(self, analyzer, phones, suggester=None):
        super().__init__()
        self.analyzer = analyzer
        self.phones = phones
        self.calls = []
        self.drafts = None
        self.suggester = suggester
        if suggester is not None:
            self.description = ("Check the original word and replacement with both Colab tools. The returned dict "
                                "has suggestion_score (the March 0.60 selection measure), word_phone_score "
                                "(a separate rhyme/length/stress check), and score (the decision value). "
                                "An unchanged word is marked identical.")

    def forward(self, word: str, compare_to: str) -> dict:
        if self.drafts is not None and not self.drafts.rounds:
            raise ValueError("Call record_drafts with your pitches and two funniest titles before checking sounds")
        word, compare_to = clean_word(word), clean_word(compare_to)
        if self.drafts is not None:
            eligible = {(clean_word(o), clean_word(r))
                        for t in self.drafts.rounds[-1]["finalists"]
                        for o, r in zip(self.drafts.original.split(), t.split())}
            if (word, compare_to) not in eligible:
                raise ValueError("Check the words of the two nominated titles. Record a revised shortlist "
                                 "before testing replacements outside those titles.")
        raw = json.loads(self.analyzer.forward(word=word, compare_to=compare_to,
                                               **({"custom_phones": self.phones} if self.phones else {})))
        similarity = raw.get("similarity") or {}
        comparison = raw.get("comparison") or {}
        observation = {
            "word": word, "compare_to": compare_to, "score": similarity.get("similarity"),
            "phones": (raw.get("analysis") or raw).get("phones"),
            "replacement_phones": (comparison.get("analysis") or comparison).get("phones"),
        }
        suggestion_raw = None
        if self.suggester is not None:
            suggestion_raw = json.loads(self.suggester.forward(
                target=word, word_list_str=json.dumps([compare_to]), min_similarity="0.0",
                **({"custom_phones": self.phones} if self.phones else {})))
            candidate = next((c for c in suggestion_raw.get("suggestions", []) if c["word"] == compare_to), {})
            observation.update(word_phone_score=observation["score"], suggestion_score=candidate.get("similarity"),
                               score_basis="identical spelling" if word == compare_to else "parody-suggestions")
            observation["score"] = 1.0 if word == compare_to else observation["suggestion_score"]
        if observation["score"] is None:
            observation["error"] = raw.get("error") or comparison.get("error") or "Unscorable comparison"
        rendered = json.dumps(observation, ensure_ascii=False)
        self.calls.append({"word": word, "compare_to": compare_to, "score": observation["score"],
                           "returned": rendered, "tool_result": raw})
        if suggestion_raw is not None:
            self.calls[-1].update(suggestion_tool_result=suggestion_raw, suggestion_score=observation["suggestion_score"],
                                  word_phone_score=observation["word_phone_score"], score_basis=observation["score_basis"])
        return observation


def parse_answer(answer):
    if isinstance(answer, str):
        answer = json.loads(answer)
    if not isinstance(answer, dict):
        raise ValueError("Pass a dictionary to final_answer")
    attempts = answer["attempts"]
    titles = [a["title"] for a in attempts]
    if len(set(t.casefold().strip() for t in titles)) != len(titles) or len(titles) < 3:
        raise ValueError("Include at least three distinct attempts")
    if answer["winner"] not in titles or answer["runner_up"] not in titles or answer["winner"] == answer["runner_up"]:
        raise ValueError("Winner and runner_up must be different titles from attempts")
    if not all(isinstance(a["why"], str) and a["why"].strip() for a in attempts) or not answer["explanation"].strip():
        raise ValueError("Explain the jokes and the winner")
    return answer


def score_from_calls(title, candidate, observed):
    aligned = len(title.split()) == len(candidate.split())
    pairs = align_swaps(title, candidate) if aligned else []
    if not aligned:
        original = list(map(clean_word, title.split()))
        replacement = list(map(clean_word, candidate.split()))
        for op, i, j, k, end in SequenceMatcher(None, original, replacement, autojunk=False).get_opcodes():
            if op == "replace":
                pairs.extend(pair for pair in observed
                             if pair[0] in original[i:j] and pair[1] in replacement[k:end])
    swaps = []
    for word, replacement in pairs:
        key = (word, replacement)
        if key not in observed:
            raise ValueError(f"Print word_phone_tool({word!r}, {replacement!r}) in a separate step, then read its result")
        swaps.append({"original": word, "replacement": replacement, "score": observed[key]["score"]})
    return {"aligned": aligned, "swaps": swaps,
            "passes": aligned and bool(swaps) and all(s["score"] is not None and s["score"] >= MIN_PHONETIC_SCORE for s in swaps)}


def run_agent(model, title, suggested, phones, analyzer, record, reference_prompt=None, humor_first=False,
              suggester=None, previous_drafts=None, revision=None):
    if humor_first and suggester is None:
        raise ValueError("The humor-first floor requires the archived parody-suggestions tool")
    phone = WordPhoneTool(analyzer, phones, suggester if humor_first else None)
    agent_tools = [phone]
    if humor_first:
        listed = suggested_pairs(suggested) if not previous_drafts else set()
        phone.drafts = RecordDrafts(title, phone.calls, listed)
        agent_tools.append(phone.drafts)
        record["draft_rounds"] = phone.drafts.rounds
        if previous_drafts:
            first = previous_drafts["draft_rounds"][0]
            if first["prior_phone_calls"] != 0:
                raise ValueError("The source slate must predate phonetic checks")
            phone.drafts.forward(first["attempts"], first["finalists"])
            phone.drafts.rounds[0].update(step=0, source="prior creative phase")
            record["prior_creative_seed"] = previous_drafts.get("seed", 42)
            record["prior_creative_steps"] = [s for s in previous_drafts["steps"] if s["step"] <= first["step"]]
    steps = []
    observed = {}
    final_evidence = {}
    step_offset = 0

    def capture(step, **kwargs):
        if not isinstance(step, ActionStep):
            return
        message = step.model_output_message
        raw = getattr(message, "raw", None)
        choice = raw.choices[0] if raw and raw.choices else None
        notes = step.observations or ""
        number = step_offset + step.step_number
        calls = [c for c in phone.calls if "step" not in c]
        for c in calls:
            c["step"] = number
            c["printed_by_agent"] = c["returned"] in notes or str(json.loads(c["returned"])) in notes
            if not c["printed_by_agent"]:
                notes += "\nword_phone_tool returned: " + c["returned"]
            c["visible_in_observation"] = True
            observed[(c["word"], c["compare_to"])] = c
        # A Python action may print only a score or assign the result to a variable.
        # Keep the complete receipt in the ActionStep that CodeAgent reads next.
        step.observations = notes
        steps.append({"step": number,
                      "reasoning": (getattr(choice.message, "reasoning_content", None) or "") if choice else "",
                      "model_output": choice.message.content if choice else step.model_output,
                      "code": step.code_action, "observations": notes,
                      "finish_reason": choice.finish_reason if choice else None,
                      "error": str(step.error) if step.error else None, "final": step.is_final_answer})
        if humor_first:
            for draft_round in phone.drafts.rounds:
                draft_round.setdefault("step", number)
            if step.model_input_messages:
                steps[-1]["input_messages"] = model._prepare_completion_kwargs(
                    step.model_input_messages, custom_role_conversions=model.custom_role_conversions,
                    convert_images_to_image_urls=True)["messages"]
        print(f"{title}: agent step {number}, {len(calls)} tool checks, final={step.is_final_answer}", flush=True)
        if step.error:
            print(f"  {str(step.error)[:300]}", flush=True)

    def recorded_answer(answer):
        if isinstance(answer, str):
            answer = json.loads(answer)
        if humor_first:
            if not phone.drafts.rounds:
                raise ValueError("Record your drafts and two funniest finalists before final_answer")
            latest = phone.drafts.rounds[-1]
            answer = {**answer, "attempts": latest["attempts"]}
            if set(latest["finalists"]) != {answer.get("winner"), answer.get("runner_up")}:
                raise ValueError(f"Choose between the two recorded finalists: {latest['finalists']}")
        return parse_answer(answer)

    def validate(answer, memory, agent):
        result = recorded_answer(answer)
        if not observed:
            raise ValueError("Print the phonetic tool results and read the observation before selecting a winner")
        if reference_prompt:
            if len(result["attempts"]) not in (3, 4) or any(
                not isinstance(a.get("humor"), (float, int)) or not 0 <= a["humor"] <= 10 for a in result["attempts"]
            ):
                raise ValueError("Include three or four attempts with numeric humor ratings out of ten")
            missing = []
            for finalist in (result["winner"], result["runner_up"]):
                if len(finalist.split()) != len(title.split()):
                    raise ValueError("Return finalists with a word-by-word alignment for the original title")
                for orig, repl in zip(title.split(), finalist.split()):
                    pair = (clean_word(orig), clean_word(repl))
                    if pair not in observed and pair not in missing:
                        missing.append(pair)
            if missing:
                action = "\n".join(f"print(word_phone_tool(word={o!r}, compare_to={r!r}))" for o, r in missing)
                raise ValueError("These finalist pairs have no prior observation. Execute these calls in one "
                                 "Python action, read the returns, then call final_answer:\n" + action)
            if humor_first:
                original = [clean_word(w) for w in title.split()]
                def passes(t):
                    return clears_phonetic_floor(original, [clean_word(w) for w in t.split()], observed, MIN_PHONETIC_SCORE)
                if (listed and not listed.intersection(title_changes(title, result["winner"]))
                        and listed.intersection(title_changes(title, result["runner_up"])) and passes(result["runner_up"])):
                    raise ValueError("The suggested-word finalist clears 0.60. Choose it, or record a funnier "
                                     "suggested-word finalist and verify its exact changes.")
                if not passes(result["winner"]):
                    if passes(result["runner_up"]):
                        raise ValueError("The proposed winner misses 0.60. Choose the eligible finalist or revise.")
                    nominated = {frozenset(d["finalists"]) for d in phone.drafts.rounds}
                    if len(nominated) < 2:
                        raise ValueError("Neither finalist clears 0.60. Revise the drafts and verify the new finalists.")
            final_evidence.update(observed)
            return True
        for a in result["attempts"]:
            assessed = score_from_calls(title, a["title"], observed)
            if a["title"] == result["winner"] and not assessed["swaps"]:
                raise ValueError("Check at least one changed word in the winning title with word_phone_tool")
        final_evidence.update(observed)
        return True

    agent = CodeAgent(tools=agent_tools, model=model, additional_authorized_imports=["json"],
                      **({"max_print_outputs_length": 12000} if humor_first else {}),
                      instructions=revision["applied_instructions"] if revision else BASE_INSTRUCTIONS,
                      step_callbacks=[capture], final_answer_checks=[validate], max_steps=7,
                      return_full_result=True, verbosity_level=LogLevel.OFF)
    task = f"""Create a funny parody of the movie title "{title}". Our audience likes absurd, dark or adult wordplay.
Start with a single-word swap. Then try a two-word change; keep more of the original if it sounds funnier.
Compare at least three distinct pitches. Choose the funniest whole phrase that still sounds recognizable.
Suggestions from parody-suggestions are optional: {json.dumps(suggested, ensure_ascii=False)}

Use Python code in the agent's code block. For each changed word of each attempt, call
print(word_phone_tool(word="original", compare_to="replacement")) with actual words.
Read the returned observations in your next step. A null score is a real result. You can revise, test new swaps,
and reconsider. A high score can also mean a rhyme. Historical raw scores can exceed 1. Use the sound and the premise to decide.
After checking the words, call final_answer with a dictionary in this shape:
{{"attempts":[{{"title":"...","why":"the comic premise"}}], "winner":"exact title", "runner_up":"exact title",
"explanation":"why this wins, what changed and what you kept"}}. The pipeline attaches the actual scores."""
    if reference_prompt:
        record["reference_prompt"] = reference_prompt
        task = reference_prompt
        if humor_first:
            task = task.replace("scores > 0.8", "scores >= 0.6").replace("scores > 0.6", "scores >= 0.6")
            task += """

Working with CodeAgent:
First call record_drafts(attempts, finalists) with three or four dicts {"title": ..., "humor": 8, "why": ...}
and the two titles you personally find funniest. Save the full premises before making any verifier call.
Go for the uncomfortable, dirty or dark double meaning when it makes a better joke. Give the image or premise
that made you laugh. A proposed title must mean something to the audience. Explain what the changed phrase
means and the scene it suggests. Discard a respelling if its only appeal is how it sounds.
Read the returned words for each original word. Try several of those words on the movie; ask which one
turns its best-known scene, character or central line into an outrageous double meaning you hear in the title.
Compare the jokes before looking at scores. Nominate your funniest supplied-word pitch and the funniest
remaining aligned pitch. Prefer a qualifying nominee using a supplied change. If there are no suggestions,
invent your own; if every suggestion falls flat, pass skip_reason to record_drafts naming an actual returned
word and why it fails comically. Check the exact spelling: a neighboring word's score cannot verify your swap.
For each word of both nominees, execute print(word_phone_tool(word=original, compare_to=replacement)).
Use a Python loop over zip(original_title.split(), finalist.split()). Only test these nominees; do not
search over cross-products of words. The phonetic tool returns a dict.
Read the real observations on the next step. 0.60 is the inclusive floor; once a pair clears it, extra
phonetic similarity earns no preference. Choose by humor. If both miss, record revised pitches and check again.
If a second, different pair still misses, stop searching. Report your preferred fallback and explain that
there is no qualifying winner. The report flags this result and keeps it out of training candidates.
Your report attaches the recorded attempts and all scores automatically.
Finish with final_answer({"winner":"exact finalist title", "runner_up":"other finalist", "explanation":"short reason"}).
Use the thinking channel for your creative comparisons and reconsideration after verification."""
            if suggester is not None:
                task += """

Scoring detail: the supplied candidate similarities are from parody-suggestions. For every proposed
replacement, word_phone_tool executes that same scoring code on the pair and separately runs word-phone.
Use suggestion_score for the inclusive 0.60 floor on changed words. An unchanged word is an identity.
The separate word_phone_score can be 1.0 for a rhyme: do not use it to substitute for the candidate similarity.
Quote both by name if giving numbers. Once a change clears 0.60, choose the funniest whole phrase."""
            if previous_drafts:
                task += ("\nYour initial drafts and nominations were already recorded before any phonetic checks. "
                         "Use these original pitches to start verification. Do not record a duplicate slate. "
                         "If both fail the floor or you need a better eligible alternative, revise and check again.\n"
                         + json.dumps({"attempts": first["attempts"], "finalists": first["finalists"]}, ensure_ascii=False))
        else:
            task += """

CodeAgent execution and return format:
Use the provided word_phone_tool directly. Call word_phone_tool(word=ORIGINAL, compare_to=REPLACEMENT),
in that order. It returns a Python dict; check["score"] is the numeric field. Print each result,
then read the observation in a later step. Check every word, including unchanged words, in both finalists.
The suggested scores are from a different tool. Use only the observed verifier scores as verification.
Aim above 0.8 and try alternatives for results below 0.6, as described above. Explain any shortfall.
The report attaches the complete verification table to your final answer. Do not print a second report
or recompute a minimum. Once you have seen the checks, call final_answer in one short Python action.
Return final_answer with a dictionary: {"attempts":[{"title":"...","humor":8,"why":"..."}],
"winner":"exact title","runner_up":"exact title","explanation":"why the final phrase works"}.
Use the thinking channel for brainstorming. Keep executable actions in the CodeAgent code block."""
    if revision:
        record.update(guidance_sha256=revision["feedback_sha256"],
                      applied_instructions_sha256=revision["applied_instructions_sha256"])
        if revision["task_comments"]:
            task += "\n\nThe human reviewer's comments behind the creative direction:\n" + "\n\n".join(revision["task_comments"])
    record.update(task=task, steps=steps, tool_calls=phone.calls)
    result = agent.run(task)
    if result.state == "max_steps_error" and observed:
        # Keep the real observations and the model's own drafts in agent memory.
        # A framework-generated fallback alone is never counted as a checked result.
        record["initial_state"] = result.state
        step_offset = max(s["step"] for s in steps)
        continuation = ("Finish the same movie title using your own drafted attempts and the tool observations "
                        "already in memory. If finalist pairs are missing, check them together in one Python action. "
                        "Then call final_answer once with attempts (title, humor, why), winner, runner_up, explanation. "
                        "Use a short literal Python dictionary; the report will attach the observed scores.")
        record["continuation_task"] = continuation
        result = agent.run(continuation, reset=False, max_steps=3)
    record["state"] = result.state
    if result.state != "success" or not any(step["final"] for step in steps):
        raise RuntimeError(f"CodeAgent did not reach a checked final_answer ({result.state})")
    answer = recorded_answer(result.output)
    finalists = (answer["winner"], answer["runner_up"])
    attempts = []
    for a in answer["attempts"]:
        if reference_prompt and a["title"] not in finalists:
            attempts.append({**a, "finalist": False})
        else:
            attempts.append({**a, **score_from_calls(title, a["title"], final_evidence), "finalist": a["title"] in finalists})
    record.update(parody=answer["winner"], runner_up=answer["runner_up"], explanation=answer["explanation"],
                  attempts=attempts)
    if reference_prompt:
        fields = ("word", "compare_to", "score", "step")
        if humor_first and suggester is not None:
            fields += ("suggestion_score", "word_phone_score", "score_basis")
        record["finalist_checks"] = {
            t: [{k: final_evidence[(clean_word(o), clean_word(r))][k] for k in fields}
                for o, r in zip(title.split(), t.split())] for t in finalists}
    winner = next(a for a in record["attempts"] if a["title"] == record["parody"])
    record["passes_threshold"] = winner["passes"]
    if humor_first:
        record["passes_threshold"] = winner["passes"] and all(
            c["score"] is not None and c["score"] >= MIN_PHONETIC_SCORE for c in record["finalist_checks"][record["parody"]])
        record["qualifying_parody"] = record["parody"] if record["passes_threshold"] else None
        record["fallback"] = None if record["passes_threshold"] else record["parody"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("llama-server", "base-model", "titles", "output"):
        parser.add_argument("--" + name, required=True)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--word-bank", type=Path)
    sources.add_argument("--colab-notebook", type=Path)
    parser.add_argument("--example-notebook", type=Path, help="Running Man notebook, used with --word-bank")
    parser.add_argument("--humor-first", action="store_true")
    parser.add_argument("--drafts-from", type=Path, help="Reuse pre-check creative slates where present")
    parser.add_argument("--guidance-plan", type=Path, help="Prepared human feedback from the RunPod launcher")
    parser.add_argument("--limit", type=int, required=True)
    args = parser.parse_args()
    if args.guidance_plan and (not args.humor_first or args.drafts_from):
        parser.error("--guidance-plan requires humor-first generation with fresh drafts")
    if args.example_notebook and not args.word_bank:
        parser.error("--example-notebook requires the March --word-bank")
    if args.humor_first:
        if not args.word_bank or hashlib.sha256(args.word_bank.read_bytes()).hexdigest() != BANK_SHA256:
            parser.error(f"--humor-first requires the pinned 2025 word bank: {BANK_URL}")
        args.example_notebook = args.example_notebook or NOTEBOOK
    if args.drafts_from and not args.humor_first:
        parser.error("--drafts-from requires --humor-first")
    previous = {}
    if args.drafts_from:
        with args.drafts_from.open(encoding="utf-8") as source:
            for line in source:
                row = json.loads(line)
                if row.get("draft_rounds"):
                    if row.get("model") != args.base_model or row.get("reference_cell") != HUMOR_BATCH_CELL:
                        parser.error("Cached creative drafts must use the same model and Colab cell")
                    previous[row["input_title"]] = row
    if args.colab_notebook:
        words, phones, cutoff = notebook_words(args.colab_notebook), {}, 0.8
    else:
        words, phones = load_word_bank(args.word_bank)
        cutoff = MIN_PHONETIC_SCORE
    titles = read_titles(args.titles)[:args.limit]
    if not titles:
        raise SystemExit("No titles to generate")
    print(f"CodeAgent: {len(words)} words, {len(phones)} custom pronunciations, cutoff {cutoff}", flush=True)
    revisions = COLAB_TOOLS if args.colab_notebook else RUNNING_MAN_TOOLS
    if args.colab_notebook or args.example_notebook:
        suggester, analyzer = [load_tool(repo, revision=rev, trust_remote_code=True) for repo, rev in revisions.items()]
    else:
        suggester, analyzer = load_parody_tools()
    model = ReasoningGGUFModel(model_id=args.base_model, api_base=args.llama_server.rstrip("/") + "/v1",
                        api_key=os.environ.get("LLAMA_API_KEY") or "local", flatten_messages_as_text=True,
                        client_kwargs={"timeout": 120, "max_retries": 1}, temperature=0.7, top_p=0.9,
                        max_tokens=3072, seed=42, extra_body={"repeat_penalty": 1.1,
                        "chat_template_kwargs": {"enable_thinking": True}})

    output_dir = Path(args.output).parent
    revision = None
    if args.guidance_plan:
        plan = json.loads(args.guidance_plan.read_text(encoding="utf-8"))
        revision = revise_prompt(model, plan)
        (output_dir / "prompt-revision.json").write_text(json.dumps(revision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def generate_title(title, seed):
        model.kwargs["seed"] = seed
        row = {"input_title": title, "model": args.base_model, "agent": "smolagents.CodeAgent",
               "suggestion_cutoff": cutoff, "word_count": len(words), "custom_pronunciations": len(phones)}
        if args.colab_notebook or args.example_notebook:
            row["tool_revisions"] = revisions
        try:
            suggested, errors = suggestions_for(title, suggester, json.dumps(words), phones, cutoff)
            row.update(suggested_words=compact_suggestions(suggested, top_n=len(words)),
                       suggestion_tool_results=suggested, suggestion_warnings=errors)
            prompt = None
            if args.example_notebook:
                row["reference_cell"] = HUMOR_BATCH_CELL if args.humor_first else RUNNING_MAN_CELL
                prompt_suggestions = row["suggested_words"] if args.humor_first else suggested
                prompt = running_man_prompt(args.example_notebook, title,
                                            {w: prompt_suggestions[clean_word(w)] for w in title.split()}, row["reference_cell"])
            prior = previous.get(title)
            if prior and prior.get("suggestion_tool_results") != suggested:
                row["draft_source_note"] = "Source suggestions differ; generated fresh pitches."
                prior = None
            run_agent(model, title, suggested, phones, analyzer, row, prompt, args.humor_first, suggester, prior, revision)
            if not any(s["reasoning"].strip() for s in row["steps"]):
                raise ValueError("No model reasoning trace was captured")
        except Exception as exc:
            row["error"] = str(exc)
        row["seed"] = seed
        return row

    failed = False
    with Path(args.output).open("w", encoding="utf-8") as out:
        for title in titles:
            row = generate_title(title, 42)
            if args.humor_first and "error" in row:
                print(f"{title}: retrying failed agent run ({row['error']})", flush=True)
                first = row
                row = generate_title(title, 43)
                row["prior_runs"] = [first]
            failed |= "error" in row
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            label = row.get("parody", row.get("error"))
            if row.get("fallback"):
                label = f"no qualifying winner (fallback: {row['fallback']})"
            print(f"{title} → {label}", flush=True)
    if args.humor_first:
        write_transcripts(Path(args.output), output_dir)
    if revision:
        write_next_feedback(output_dir, revision)
    if failed:
        raise SystemExit("One or more CodeAgent runs failed; inspect results.jsonl")


if __name__ == "__main__":
    main()

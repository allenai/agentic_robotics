# CLAUDE.md

Guidance for AI coding agents working in `agentic_robotics`. This file is loaded automatically. It
says what the code is, which rules hold while you work in it, and why each rule is right. Other
docs point at it rather than restating it. It describes the repository as it is: git holds how it
got here.

## What this project is

An **MCP-orchestrated agentic loop over black-box robot skills, with per-domain adapters**. A
vision-language model is handed a set of MCP tools, each one a robot *skill*, and drives a task by
choosing tool calls. There is no verifier and no success signal: the agent sees the post-state
image and decides for itself what happened. The output worth having is an account of **how the
loop fails**, not a success rate.

The design commitment is the seam. `core/` is the MCP host and knows nothing about any robot or
simulator; each domain is an MCP server behind an install extra. Only JSON-serializable values
cross the seam: no live Python objects, no tensors, no environment handles. `core` never imports
`blocksuite`, `torch` or `isaaclab`, and the two-interpreter layout is what enforces it, since
`core` runs in an interpreter where none of those is installed.

### `core/`, which knows no robot

- **`agentic_loop.py`**: `AgenticLoop.enter_loop()` runs the model↔tool loop up to a
  `max_tool_calls` budget, re-injecting an instruction with the remaining count each turn, stopping
  when the model calls no tool, then taking one final-answer turn.
- **`claude_code_loop.py`**: the same loop hosted by Claude Code through the Claude Agent SDK
  instead of by this repo, with the loop's bookkeeping (per-call state snapshots, the cutoff, the
  status line) supplied through the SDK's hooks.
- **`mcp_conversion.py`**: MCP tool definitions and results to and from the OpenAI Responses API,
  text and base64 images.
- **`foundation_models/`**: `ModelClient` (a `Protocol`) and `OpenAIModelClient`, a thin
  `AsyncOpenAI` wrapper over the Responses API. OpenRouter by default (`OPENROUTER_API_KEY`), and
  any OpenAI-compatible endpoint through `base_url`, so a locally served model needs no second
  client: `examples/mcp_image.py` drives one through Ollama's `/v1/responses`. `Transcript` logs
  every conversation item to `transcript.jsonl`, spilling inline images to `images/`.
- **`images.py`**: `Resolution`, raw `PixelCoord`, and `ImagePoint`, normalized to `[0,1000)`, the
  coordinate space the model sees. That space is finer than any render grid, so
  `ImagePoint → PixelCoord → ImagePoint` is lossy; never echo a converted point back to the model
  as if it were what the model typed.
- **`cameras.py`**: `CameraModel`, a pinhole camera with intrinsics and world-frame extrinsics;
  `points_to_pixels` (NaN behind the camera), `pixel_to_ray`, `projected_ray_length_px`.
- **`states.py`**: the object-centric simulator state as pydantic models, so it crosses MCP.
- **`transforms.py`**, **`filesystem.py`**, **`logging.py`**, **`runtime_env_checks.py`**: a 4x4
  from position and rotation; unique run directories and video and image export; one handler on
  the `agentic_robotics` namespace; a record of the interpreter, packages and git state a run
  started under.

### `adapters/toyblocks/`, the one domain: ToyBlocks on Isaac Sim

- **`cli.py`** is the entry point, installed as `agentic-robotics`
  (`python -m agentic_robotics.adapters.toyblocks.cli` is the same command). Every supported path
  through the repository is a mode of it: `episode`, `human`, `page`, `menu`, `mock`, `portal`,
  `generate-env`, `serve`, `tasks`, `leap`, `pool`. It re-executes itself under an interpreter that
  can import ToyBlocks, so either interpreter can launch it.
- **`config.py`** is the schema only: `ToyBlocksConfig`, a frozen dataclass of one run's
  parameters, with `OracleConfig`, `SkillBackend` and the task table. Values come from an
  experiment file, `experiments/<name>.toml`, read by **`experiment.py`** and chosen with
  `--experiment`; a flag overrides a value for one run; the resolved configuration is written into
  the run directory as `config.json`, so a result is reproducible from its artifacts alone. No
  module holds mutable settings that another module patches. **`maniflow_config.py`** names one
  `ManiFlowConfig` per policy checkpoint.
- **`toyblocks_env.py`** wraps the Isaac Sim gym environment. **Isaac Sim launches at import
  time**, and the imports below that launch must stay below it (`# ruff: noqa: E402`).
- **`toyblocks_runner.py`** is the threading model. Isaac must be stepped from the thread that
  created it, so the simulator owns the main thread, the agentic loop runs on a second one, and
  `SimulatorDispatcher` hands skill coroutines back to the simulator thread through a queue. Do not
  restructure this: stepping PhysX from the wrong thread crashes the process rather than raising.
- **`runner_bootstrap.py`** is the boot ordering, the run directory and the episode harness that
  every mode shares. Generated task packages live in this repository, under `data/tasks/`, and
  import as `agentic_robotics.tasks.<slug>` through the `tasks` package's extended path; ToyBlocks's
  own tree is never written to.
- **`toyblocks_mcp_server.py`** is the abstract base: the skill surface as MCP tools, frames
  recorded to an mp4. **`oracle_mcp_server.py`** declares the scripted surface, nine tools
  (`get_zoomed_image`, `pick`, `stack`, `place_on_table`, `rotate`, `flip`, `return_home`,
  `open_gripper`, `close_gripper`), every body abstract. **`maniflow_mcp_server.py`** drives a
  served ManiFlow policy over a websocket. The two surfaces differ on purpose.
- **`skills/`** implements them. `inline_oracle.py` is the scripted skills, planning with cuRobo
  against the live Isaac Lab scene; `rotate` and `flip` run small RL policies through
  `rl_executors.py` under the PD gains in `stiffness_profiles.py`, and `ik.py` is the seeded
  inverse kinematics the short segments use. `quirk_skills.py` is the same module with the LEAP
  deficiencies layered on, kept separate so the scripted skills stay the ground truth they are
  measured against. `maniflow_skills.py` is the learned-policy side.
- **Around the boot**: `task_variants.py`; the checks that refuse to start a run on a scene the
  model cannot act in (`block_asset_check.py`, `scene_preflight.py`, `reset_state_check.py`);
  `block_pointing.py` (which block a pointed pixel names); `human_model_client.py`; `portal.py`,
  the run browser, stdlib only so it runs under either interpreter; `web_ui/`, the page a human
  drives or watches from.
- **The experiments**: `leap/` (the experimentation phase, the pool of served simulators and the
  evaluation), `release/` (the Search & Spell scoring and families, the blocksworld modes and their
  prompts), `tasks/` (the task tables), and `drivers/` (the suite and build drivers, each a console
  script named in `pyproject.toml`).

### The model's interface is code

Tool names, parameter schemas, tool docstrings and the text of errors raised inside a skill are the
interface the model programs against. They are **behavior, not documentation**: editing them changes
what runs, and deserves the scrutiny of a code change. A skill's docstring is a prompt, so
preconditions and units belong in it.

Anything echoed back to the model must be what the model actually sent. A point that has been
through the pixel grid and back is not; an agent told it asked for a slightly different point than
it typed will spend the episode hunting the difference instead of the block.

**Documentation about one file belongs in that file.** A design note in a directory beside the
code drifts from what it describes; write it where the reader of that code will hit it.

The sibling ToyBlocks repository (whose package imports as `blocksuite`), and its submodules, is outside this repository's authorship
rules; see "Commit your own work".

---

# Authorship

"The developer" is the developer leading the session. An explicit request from the developer to
bend one of these rules wins over this file for that request.

What the repository is for, in priority order when they conflict: a core interface for agentic
systems reusable across its developers' research careers; fast iteration to new results through
domain adapters; and a framework others can use. Reusability first, then results, then code
quality.

## Edit whatever the task needs, with four exceptions by kind of content

There are no developer-owned and no agent-owned directories. The exceptions are kinds of text, not
places:

- **`README.md` and `LICENSE`**: discuss first and edit only with explicit permission. The README is
  the repository's voice to its users. Offer the change in a fenced block instead.
- **Agent-facing text**: a prompt, a tool docstring, a parameter description, the text of an error
  a skill raises. Whenever its phrasing could affect agent behavior, exit conditions or an
  experiment's outcome, the developer chooses the words. Propose them in chat; write them into the
  file once given. The reason is above: that text is behavior.
- **Paper-visible text**: the developer writes anything that will appear in a paper or its
  appendices.
- **Default values**: the developer chooses the default of any parameter important enough to
  describe in a paper's appendix. Propose; do not set.

Configuration files, `pyproject.toml` and this file among them, are yours to edit when the task
calls for it.

## Ask before anything that touches the experiments

A missed escalation costs far more than an unneeded one. An unneeded question costs a turn of
waiting; a choice made without the developer's knowledge can change experimental outcomes and
invalidate results. So make no choice on your own that directly affects experimental setup or
outcomes: a termination policy or step cap, a scorer's threshold, which checkpoint or scenes a
comparison uses, a logging schema a result will be read from, a prompt, a default hyperparameter,
the name of a research unit. Say in one line what the choice is and what it would settle, then
wait. Draft the mechanism freely; the decision is the developer's.

A flag is not a one-time act. When the thing you flagged changes, re-check the flag: a callout that
describes a distinction the code no longer makes is worse than none. A variant no run has ever
exercised is deleted, not maintained.

## Co-design, then implement against a specification

The developer's possible levels of involvement, from most to least effort: writing the tests by
hand, implementing by hand, rewriting agent code by hand, line-by-line review, co-design before
implementation, contract by specification, implementation from informal conversation. Rewriting by
hand costs more than review and exercises nothing, so expect co-design and review: go into the
intended design, its assumptions and its class structure in conversation before implementing, and
expect core interfaces to be reviewed rather than rewritten.

A **specification** is a developer-written, or explicitly developer-approved, description of an
intended design or functionality. You may draft one; it is not canonical until approved. Code that
has a correct answer which can be verified, programmatically or visually, is never implemented by
hand by the developer: implement it, with its verification, against the approved specification.
The approved specification is the request for that verification code.

## Verification, not speculative tests

Write no speculative test, smoke test, or "let me just check it works" harness. Verification is
written in two cases. First, the code has a correct answer and a developer-approved specification
exists: the verification is part of the deliverable. Second, a real failure was observed, fixes were
tried against it, and the check that caught it is kept, whether as an external check or as an
in-source guard that refuses to start. If a test would be valuable beyond those two cases, say so
and ask first. When you want evidence for a claim you are about to make, either ask or make a weaker
claim.

## Surprising results get checked before they are explained

Before rationalizing a surprising result to the developer, confirm as far as you can that the result
is real: re-read the per-trial fields, check the instrument against a known answer, look at the
frames. A worse score is as often a step cap, a scorer bug or a probe's start state as a finding.
Before a surprising result goes to another party, a developer visually verifies the anomalous
behavior, so point at the artifact (the clip, the image, the record), not at the number.

## Keep the developer's reading current

The developer reads the top-level docstring and the module-level class docstrings of every
agent-implemented source file, every script important enough for the README, and every script that
produces reported results, and rereads them when they change. When a task changes a file enough
that one of those docstrings had to change, end the task by naming the files and docstrings to
reread.

## Fix at the source

A defect in a file of this repository is fixed where it is, in the turn you find it, and the
summary says so. A workaround is two things to maintain, the shim and the claim about why it
exists, and the claim is the one that goes stale. Workarounds are for the parent repository only,
where an agent patches but does not own. Offering to fix a defect instead of fixing it is the same
failure: it spends a developer turn re-granting a permission this file already gives. The one-line
callout is reserved for `README.md`, `LICENSE`, agent-facing text, the parent repository, decisions
that touch the experiments, and anything that starts a long or resource-hungry run.

## Extend the supported path; never copy it

Every agentic run goes through `agentic-robotics`. A new kind of run is a new mode of it, sharing
`prepare_scene`, the policy-server and block checks, `runner_bootstrap.run_episode` and the
manifest, with whatever differs supplied as a thread function or a server factory. One level down
the same holds: a script that needs what `runner_bootstrap` or an adapter already does calls it, and
extends it with a parameter when it falls short.

A copy of the boot looks harmless because it works on the day it is written. What it loses is every
check the original grows afterwards, and every later fix lands in one place and not the other; a
suite can then run on a task package built from the wrong library with no sign until the results
come back wrong. The bypass always costs more than the mode would have.

A local copy is still right for a **probe**: a one-off that reads a package, drives a served
simulator over MCP or renders a scene, and never runs an agent. Say in its docstring that it is one
and what it bypasses, so a reader knows it carries none of the checks.

An example may not carry its own copy of something `core/` or an adapter already provides; import
it. When an example needs a capability the codebase lacks, write it where it belongs, which may be
`src/`, after the design conversation above. When reuse is genuinely worse than a local copy, keep
the copy and say why.

## Where things live

| Location | Purpose |
| --- | --- |
| `src/` | The library: `core/`, the adapter, the `agentic-robotics` CLI and the suite drivers it runs |
| `examples/` | A no-simulator example, and under `agent/` the probes, prototypes and task generators; nothing here is imported by the CLI |
| `scripts/` | `openrouter_spend.sh`, the budget check; `run_suite_local.sh`, an evaluation suite on one machine |
| `data/` | PDDL problems, generated task packages (`tasks/<slug>/`), block-face maps, checkpoint block-asset records, the robot description, example states. Removing a package costs an Isaac boot to rebuild, so say when you do |
| `tests/` | `pytest` |

When a doc contradicts the code or another doc, say so and ask which is intended rather than
guessing.

## The tooling pass

Ruff formats and lints this repository. At the end of a task, on the files you changed and only
those, run `ruff format` once and `ruff check` once; rerun only to confirm a fix, never to look for
more. Under `src/`, fix every finding in that one pass. `pyproject.toml` carries a relaxed ruleset
through per-file ignores for `examples/`, `scripts/`, `tests/` and the drivers; fix only the
findings that are real defects there and leave the rest. The pinned Ruff runs without the project
environment: `uvx ruff==0.16.2 format <files>` and `uvx ruff==0.16.2 check <files>`.

That pass is the whole of the polish. Re-reading your own work for correctness is wise: that it does
what was asked, that it will run, that it matches the signatures it calls. Re-reading to find
something to improve is not. "Is this correct?" is always worth a tool call; "could this be
tidier?" never is. Annotate types within reason and move on; a bare `Any` or an unparameterized
generic is fine where the checker allows it. No greps for style precedent, no reading a second
file to copy a convention.

## Commit your own work

When a batch of edits forms a sensible, atomic commit, commit it, staging only the files you
edited, by name, never the whole tree, with a real message saying what changed and why; then push
the branch, provided the remote has nothing the local branch lacks. If it does, stop and tell the
developer rather than merging or rebasing. Do not run `git merge`, `git rebase` or anything that
rewrites history. Do not commit a developer's uncommitted edits: an edit you did not make is not
yours to commit. The parent repository is outside these rules: patch ToyBlocks when a cross-repo
fix needs it, and do not commit or push it unless a developer asks.

## Other developers and their agents

These are the default expectations for any developer contributing here, and their agent follows
them the same way. Their purpose is that the developer leading a session is aware whenever their
agent makes a change that could affect results or users. How another developer's changes are
reviewed, and whether an agent records a developer's override of a rule in the commit message, are
not settled; ask before assuming either.

## Example

**Task:** wire episode setup for a new recorder and a viewer refresh into the ToyBlocks adapter.

**Wrong:** draft it under `examples/agent/`, report that it wants promoting, and wait for a
developer to migrate it line by line into `toyblocks_env.py`.

**Right:** agree the design in conversation, implement it in `toyblocks_env.py` with the
verification the agreed specification calls for, run the tooling pass on the files changed, commit,
and end by naming the docstrings the developer should reread. Ask first about any part of it that
would change what an experiment measures.

**Wrong:** edit `README.md` with new clone instructions. **Right:** offer the bash block in chat.

---

# Agent conduct

The part above governs *what* an agent writes and *when it asks*. This one governs *when it stops*
and *how it answers*.

## Answer the question that was actually asked

A yes/no question gets **"Yes."** or **"No."** as the first word, before any elaboration. A
question about status is not a request for work, a summary or a plan. If the honest answer is "no,
because X remains", say that in one sentence. Never respond to a direct question by doing more work
first. Answer, then offer.

### Reading is not unrequested work

When a question needs a file, a transcript, a run artifact or a git log to answer, **read it and
answer**. Do not offer to read it; "say if you want me to look" turns one turn into three and
charges the asker for a lookup they already asked for by asking. The rules below about not starting
work you were not asked for are about *writing* and about *runs*; git reads, greps, `--help`, a
syntax check, a compile and reading a finished run's artifacts off disk are always fine, and asking
permission for one of those is the failure, not the caution.

A question whose answer is in the tree gets the answer, never a hypothesis. Whether a reset
restores the gripper, whether a channel drops messages, whether a warmup can be deferred: each is
one grep away, and a guess offered where a read was available is worse than no answer, because it
sounds like a finding.

## Use the repo's own words

Use the established name for a thing: the repo's, the library's, or the one the developer just
used. Do not invent a label for something that already has one. An invented name forces the reader
to infer what you mean or spend a turn asking, and cannot be looked up anywhere. The same goes for
invented indexes: numbering a set of defects across two files couples the files to each other, and
the numbering is wrong in both as soon as one defect is fixed. Describe the thing where it is, in
the words the tree already uses.

A research concept with no name yet is the high-stakes case, because the name becomes a unit the
experiments are counted in and spreads through the tree, the docs and the chat before a developer
has seen it. Naming a research unit is a research decision (see "Ask before anything that touches
the experiments"): propose the term, say that it is a proposal, and use the developer's word once
given. The fixed terms for the LEAP evaluation suite: a **task family** is one of the eight factor
combinations; a **variant** is an abstract PDDL-level variation within a family (three per feasible
family, nine per outside family); a **task instance** is one continuous instantiation of a variant
(32 sampled per variant).

### Stock phrases that assert importance

Write what breaks, not that something is important. "This is load-bearing", "this is critical",
"note that", "it is worth emphasizing": each announces that a claim matters instead of saying what
it does, and the sentence is almost always shorter and more useful without it. The tell is
structural: the phrase is followed by a colon or a dash and then the actual reason. `Ordering here
is load-bearing: the config patch has to land before anything imports the adapter` says one thing
twice, and only the second half can be acted on. Delete the left side.

Expect some phrase not listed here to be doing the same job. These accumulate the same way every
time: a phrase gets used once where it fits, reads as competent, becomes the default marker for
"pay attention here", and because an agent reads its own output back out of this tree, every use
makes the next one likelier. When one shows up across unrelated files, state the effect and delete
the label. An agent has no reliable access to why it writes a phrase, so an explanation it offers
for one is a hypothesis, not a report; the distribution is greppable, check that instead. And a
grep across agent-written prose can confirm a habit but never justify one: "the repo's own words"
means the developers' words, `README.md` and what the developer says in chat.

## Design history belongs in commits

A docstring, a comment, a rule in this file or a config file describes the code as it is now and
why that is right. It never records what the code used to do, when it changed, who decided it or
where it was moved from: no "until <date> this was", no "revised on <date>", no "moved here from",
no "(chat, <date>)" after a rule, no "History:" paragraph, no developer's name. Git holds all of
that, exactly and for free: `git log` and `git blame` answer when and who, the commit message
answers why it changed. A second copy in the file is the one that goes stale, and every reader
after the change has to sift it to find the current rule.

What does belong is the reason the current design is right, in the present tense, and in
particular why the obvious alternative is not taken: "tool-less turns are re-prompted because the
base loop would otherwise exit on the first one" is a fact about the system, not an event. A date
stays only when it identifies a run or artifact a reader may need to find, never to say when a rule
began. Refer to people by role. Run history lives in commit messages, which may name the people in
them; the LICENSE names its copyright holder and is not a rule. When a change makes a docstring's
account of the code wrong, rewrite the account; do not append an update to it.

## Stop when the deliverable exists

Finishing the thing asked for ends the turn. Do not continue into cleanup, hardening, extra files
or "while I was in there" improvements; unrequested work costs review time and buries the thing
that was asked for. If you notice something worth doing that was not asked for, name it in one line
and let the developer decide. The exception is a defect in a file of this repository, which "Fix at
the source" says to fix in the turn you find it.

## Starting long or resource-hungry work

The agent may boot a simulator, hold the GPU and start paid runs on its own. **Anything expected to
cost more than $3 needs the developer's explicit permission beforehand**: name the command, say what
it will cost and what it would settle, and wait for the yes. Estimate the cost from the budget cap
the run is given, not from a hope that it stops early.

**Check the OpenRouter budget first.** Before starting any paid run, and once at the start of a
session, run `scripts/openrouter_spend.sh`. Two ceilings apply, the key's own limit and the
account's prepaid credit; compare the remaining credit with the run's cap plus a margin and with the
rest of the day's planned runs, and if it does not fit, say so before anything starts, since a run
that hits a ceiling dies mid-episode with HTTP 402. `agentic-robotics leap` and
`agentic-robotics-suite run` make the same check themselves and refuse to start unless
`--ignore-budget` is passed.

Isaac processes started from a session outlive it, holding GPU memory; Isaac ignores SIGTERM, so
clearing one takes SIGKILL by PID, and only the session that started it knows the PID. That is why
every such run follows the next section: the developer must be able to see what is running and
stop it without asking the agent. None of this licenses guessing at what a run would have shown.
Where a claim needs a run behind it, say plainly that it is unverified and give the command that
would verify it.

## Long-running jobs

When a run boots a simulator, spends money or takes minutes (a pilot, a render batch, an experiment
driver), whoever started it, two things hold, both so the developer can tell what is still running
without asking:

- **The process runs in a detached tmux session, and one subagent waits on it.** The tmux session
  keeps the job alive if this session drops. The subagent, named after the job, does nothing but
  block on the job's log and report its outcome; it is the visible handle in the agent list. A
  background shell loop is not a handle: the developer cannot see it, and "an agent is running"
  must never be said when none is.
- **Kill a tmux session as soon as its job is done.** Its logs are already in files; a finished
  session only pads the list. `tmux ls` is the list of what is running.

Killing a process a developer started is theirs to decide; so is one the agent started if it is
still doing paid work and the developer has seen it. A finished job's leftover process (an Isaac
close that never returns, a session holding `sleep infinity`) is the agent's to clear.

## Say what was run and what was not

"Written" and "run" are different states, and conflating them is the failure this project is built
to avoid. If you have not executed something, say so plainly and say what it would take. Never
describe untested code as working.

That rule is about the **conversation**, where a claim is made once and consumed at once. Do not
hand-maintain the same claim in a version-controlled file. Run status is a fact about the
artifacts under the runs directory; a sentence in a docstring is a cache of it, and caches go
stale. A developer will not always know to tell you they ran something, and should not have to.
Where a reader needs the confidence level at the call site, write the **judgment** and where to
check it ("the reorientation skills are unproven"), never the log ("as of last Tuesday, three
runs").

Two more classes of claim rot the same way:

- **`file:line` references in prose.** Names are stable and greppable; line numbers are not.
- **Claims about the parent repository.** These go stale invisibly, because nothing in this
  repository's history records the change. Check ToyBlocks' `git log` before repeating one.

## Always explain code changes at the end

End every turn that changed code with a short explanation: **what** changed (which files) and
**why**. Concise and factual.

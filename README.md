# agentic_robotics

An MCP-orchestrated agentic loop over black-box robot skills, with per-domain adapters.

A vision-language model is given a set of MCP tools, each one a robot skill, and drives a task by
choosing tool calls. There is no verifier and no success signal: the agent sees the post-state
image and decides for itself what happened. `core/` is the MCP host and knows nothing about any
robot or simulator; each domain is an MCP server behind an install extra. The one domain here is
[ToyBlocks](https://github.com/allenai/ToyBlocks), the block-manipulation simulator (its Python
package imports as `toyblocks`), reached through `adapters/toyblocks/`.

[`CLAUDE.md`](CLAUDE.md) describes the modules and states the authorship expectations for
agents working in this repo.

## Layout

| Location | Purpose |
| --- | --- |
| `src/agentic_robotics/core/` | The MCP host, the model client (any OpenAI-compatible endpoint; OpenRouter by default), transcripts, images, cameras |
| `src/agentic_robotics/adapters/toyblocks/` | The ToyBlocks adapter: the `agentic-robotics` CLI, the task tables, the scripted and ManiFlow skill servers, the LEAP experimentation phase, the Search & Spell scorer, and under `drivers/` the suite and build drivers behind the released experiments |
| `examples/` | A no-simulator example, `mcp_image.py` |
| `scripts/` | The tools behind the released data and figures: the PDDL problem generators (`task_gen/`), renderers and probes (`leap/`, `flip_trials.py`, `video_render.py`), the budget check and the local suite script |
| `data/` | PDDL problems, generated task packages (`data/tasks/<slug>/`, built on first use), block-face maps, checkpoint block-asset records, example states |
| `tests/` | `pytest` |

## Install

The host side needs only Python 3.11 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync                      # core + the ToyBlocks adapter's pure-Python parts
uv sync --extra dev --extra claudecode && uv run pytest tests
uv run python examples/mcp_image.py --help
```

Running a task needs ToyBlocks, which brings Isaac Sim 5.1, Isaac Lab and cuRobo v2. This repo is
ToyBlocks's `agentic_robotics/` submodule, and ToyBlocks's `agentic` install extra installs it
(editable, from the submodule) together with what its skills need from ToyBlocks's serving package;
follow the [Agentic robotics setup](https://allenai.github.io/ToyBlocks/docs/user_guide/workflows/installation_and_setup_instructions.html#agentic-robotics-setup)
on ToyBlocks's installation page (`uv pip install -e ".[dev,isaac,torch-cu128,vecgen,agentic]"` in
the ToyBlocks checkout, then the assets download). To install a checkout of this repo that is not
that submodule into an existing ToyBlocks environment instead:

```bash
uv pip install --python <that environment's python> -e ".[toyblocks]" msgpack websockets
```

The MCP SDK needs newer `starlette`, `uvicorn` and `idna` than the versions Isaac Sim pins; the
install upgrades those three, which the headless simulator does not use (they serve Isaac's
livestream and web services).

The adapter finds ToyBlocks through the installed package (or, without one, as the checkout this
repo is a submodule of). Task packages are generated with ToyBlocks's codegen on first use, into
`data/tasks/` (imported as `agentic_robotics.tasks.<slug>`, not tracked in git): pass
`--generate-env` to a run, or build a task's packages beforehand with
`agentic-robotics generate-env --task <key>`. ToyBlocks's tree is never written to.
Set `OMNI_KIT_ACCEPT_EULA=YES` for Isaac.

Episodes call the model through OpenRouter and cost money. Put the key in `OPENROUTER_API_KEY` or
in `~/.openrouter_key` (mode 600); `scripts/openrouter_spend.sh` prints the remaining budget.

Runs are written under `$AGENTIC_ROBOTICS_RUNS_DIR`, or `runs/` in this repo. The ManiFlow skills
need a policy checkpoint: `maniflow_config.py` names each relative to `$MANIFLOW_CHECKPOINT_ROOT`,
with the arm gains it was trained under. The training datasets are not needed to run one.

## Running

Every run is a mode of the `agentic-robotics` command (`src/agentic_robotics/adapters/toyblocks/cli.py`);
`--help` lists the modes and flags, and `tasks` lists the task keys and which are built on this
machine. `python -m agentic_robotics.adapters.toyblocks.cli` is the same command under any
interpreter that has the package, and a mode that needs Isaac re-executes itself under one that
can import ToyBlocks (`--python` or `$AGENTIC_ROBOTICS_PYTHON` names it).

```bash
agentic-robotics menu    --backend oracle                      # pick tasks one after another at http://127.0.0.1:8083
agentic-robotics episode --task spell_hen_patterned_0_feasible_0 --backend oracle   # one model-driven episode, scripted skills
agentic-robotics human   --task spell_hen_patterned_0_feasible_0 --backend oracle   # drive it yourself at http://127.0.0.1:8083
agentic-robotics portal                                        # browse finished runs at http://127.0.0.1:8082
agentic-robotics portal --static                               # or write them as pages under <runs dir>/portal/
```

`--backend oracle` selects the scripted skills (`skills/inline_oracle.py`): ToyBlocks's vecgen skills,
planned with cuRobo v2, which need no policy server. Without it the backend is the experiment file's `skill_backend`. The ManiFlow
skills need the policy server up first, in its own shell:

```bash
agentic-robotics serve                               # serves the checkpoint the experiment file names; leave it running

# In a second terminal, with the environment restored (activate the venv, OMNI_KIT_ACCEPT_EULA=YES,
# the OpenRouter key), from the same working directory:
agentic-robotics episode --task spell_hen_patterned_0_feasible_0
```

What a run holds fixed -- the skill backend, the model, the checkpoint, the viewer, the camera,
the seed, the scripted skills' parameters -- comes from an experiment file,
`src/agentic_robotics/adapters/toyblocks/experiments/<name>.toml`, chosen with `--experiment
<name>` (or a path); `default.toml` is what runs otherwise. `--model`, `--backend`, `--maniflow`
and `--policy-port` override it for one run, and the fully resolved configuration is written to
the run directory as `config.json`. A task whose package is not built yet needs
`--generate-env`, or `agentic-robotics generate-env --task <key>` beforehand. The pages bind to
localhost; `--lan` binds them to every interface, for a run inside a container or on another
machine. A page embeds the simulator's viewer, which listens on its own port (8080, or the next
free one) and has to be reachable from the browser too.

An evaluation suite on this machine, with the scripted skills unless a ManiFlow checkpoint is
named:

```bash
scripts/run_suite_local.sh --baseline oracle --model haiku-4.5 --suite-dir suites/local-test \
    --instances 0 --parallel 1 -- --families stack_feasible_isolated_flip
```

The experiments behind the released results:

- **Search & Spell** (`data/tasks/spell_*`): find letter blocks, some hidden, and spell a word in
  one episode. `agentic-robotics-spell-build` builds the packages, `release/search_spell_scoring.py`
  scores a final state and `agentic-robotics-spell-check` is its known-answer check,
  `agentic-robotics-spell-summary` tabulates a suite. (The drivers live in
  `src/agentic_robotics/adapters/toyblocks/drivers/`.)
- **Blocksworld goals on color cubes**: closed-loop and open-loop suites
  (`agentic-robotics-blocksworld-suite`), and the language-only plan generation in PlanBench's format
  (`agentic-robotics-planbench`, `agentic-robotics-planbench-domain`).
- **LEAP** (`agentic-robotics leap`): the experimentation phase in which the agent probes its
  skills before the task, and its evaluation suites (`agentic-robotics-suite`).

How the suites were run. Each suite (one baseline and one model, sometimes fanned out one job per
task family) and each LEAP experimentation phase was one batch job on Ai2's Beaker cluster, with one
NVIDIA A100 80 GB GPU, in a container image built from ToyBlocks with Isaac Sim, Isaac Lab and
cuRobo installed. A job started the policy server when the ManiFlow skills were in use, then ran
`agentic-robotics-suite run` (or `agentic-robotics leap` for a phase), which keeps
several simulator processes going at once (`--parallel`; each is one Isaac process of about 7.6 GB)
and appends one record per finished episode, followed by `agentic-robotics-suite summarize` on the suite
directory. The job specifications and submit scripts are specific to that cluster and are not part
of this repository; the same commands run on any machine that meets the install requirements above.

## License

Apache-2.0. See [LICENSE](LICENSE).

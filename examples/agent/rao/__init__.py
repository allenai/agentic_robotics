"""The ToyBlocks release's "Rao" experiments: Blocksworld-style tasks on color cubes, run
closed-loop and as online refinement (committed plan, pixels chosen live).

`problems` defines the task instances and their optimal plan lengths, `tasks` registers them as
`TaskConfig`s, `modes` runs one episode in one mode, and `run_suite` schedules episodes. What
the agent reads lives in `src` (`adapters/toyblocks/release/rao_prompts.py`).
"""

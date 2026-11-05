# GENERATED CODE
import gymnasium as gym

gym.register(
    id="stack_feasible_isolated_flip_02-v0",
    entry_point=f"{__name__}.env:stack_feasible_isolated_flip_02_Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:stack_feasible_isolated_flip_02_EnvCfg",
    },
)

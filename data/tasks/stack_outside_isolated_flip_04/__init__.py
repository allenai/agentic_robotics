# GENERATED CODE
import gymnasium as gym

gym.register(
    id="stack_outside_isolated_flip_04-v0",
    entry_point=f"{__name__}.env:stack_outside_isolated_flip_04_Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:stack_outside_isolated_flip_04_EnvCfg",
    },
)

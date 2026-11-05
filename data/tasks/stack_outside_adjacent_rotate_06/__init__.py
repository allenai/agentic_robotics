# GENERATED CODE
import gymnasium as gym

gym.register(
    id="stack_outside_adjacent_rotate_06-v0",
    entry_point=f"{__name__}.env:stack_outside_adjacent_rotate_06_Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:stack_outside_adjacent_rotate_06_EnvCfg",
    },
)

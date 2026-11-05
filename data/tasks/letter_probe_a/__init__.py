
# GENERATED CODE
import gymnasium as gym

gym.register(
    id="letter_probe_a-v0",
    entry_point=f"{__name__}.env:letter_probe_a_Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:letter_probe_a_EnvCfg",
    },
)

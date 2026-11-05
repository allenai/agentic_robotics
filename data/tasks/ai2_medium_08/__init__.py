# GENERATED CODE
import gymnasium as gym

gym.register(
    id="ai2_medium_08-v0",
    entry_point=f"{__name__}.env:ai2_medium_08_Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:ai2_medium_08_EnvCfg",
    },
)

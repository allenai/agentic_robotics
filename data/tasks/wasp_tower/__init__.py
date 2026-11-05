# GENERATED CODE
import gymnasium as gym

gym.register(
    id="wasp_tower-v0",
    entry_point=f"{__name__}.env:wasp_tower_Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:wasp_tower_EnvCfg",
    },
)

# GENERATED CODE
import gymnasium as gym

gym.register(
    id="find_r-v0",
    entry_point=f"{__name__}.env:find_r_Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:find_r_EnvCfg",
    },
)

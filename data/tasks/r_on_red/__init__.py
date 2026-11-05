# GENERATED CODE
import gymnasium as gym

gym.register(
    id="r_on_red-v0",
    entry_point=f"{__name__}.env:r_on_red_Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:r_on_red_EnvCfg",
    },
)

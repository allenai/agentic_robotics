
# GENERATED CODE
import gymnasium as gym

gym.register(
    id="spell_check_00-v0",
    entry_point=f"{__name__}.env:spell_check_00_Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:spell_check_00_EnvCfg",
    },
)

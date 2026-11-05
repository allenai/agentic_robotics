"""Define a data structure to represent a configuration for a ManiFlow policy."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class PDGains:
    """PD gains for the seven arm joints, as the ManiFlow policy server applies them."""

    stiffness: tuple[float, ...]
    damping: tuple[float, ...]
    effort: tuple[float, ...]

    def as_dict(self) -> dict[str, list[float]]:
        """The gains in the form ToyBlocks' `PDModeController` takes."""
        return {
            "stiffness": list(self.stiffness),
            "damping": list(self.damping),
            "effort": list(self.effort),
        }


TRAINING_PD_GAINS = PDGains(
    stiffness=(300.0, 300.0, 300.0, 300.0, 150.0, 100.0, 50.0),
    damping=(35.0, 35.0, 35.0, 35.0, 20.0, 15.0, 10.0),
    effort=(87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0),
)
"""The arm gains every released checkpoint was trained under, in both PD modes: each checkpoint's
`stats.json` records these columns with the minimum equal to the maximum on every joint. The same
numbers are the "medium" profile in `skills/stiffness_profiles.py`."""


@dataclass(frozen=True)
class ManiFlowConfig:
    """Configuration for a ManiFlow policy server."""

    checkpoint_dir: Path
    safetensors_relative_path: str
    """Path to the ManiFlow policy's safetensors file, relative to the checkpoint directory."""

    observation_mode: Literal["state+rgb", "state+rgb+depth"]
    """Observation mode requested from the ToyBlocks environment."""

    dataset_dirs: tuple[Path, ...] = ()
    """The LeRobot dataset(s) the policy was trained on, as provenance; nothing in a run reads
    them."""

    stiff_gains: PDGains = TRAINING_PD_GAINS
    compliant_gains: PDGains = TRAINING_PD_GAINS
    """The gains the policy server switches the arm between, as it was trained."""

    target_sigma: float = 0.8
    """Width of the Gaussian target heatmap used to condition the ManiFlow policy.
        Corresponds to the DEFAULT_TARGET_SIGMA used during training."""

    gripper_steps: int = 6

    policy_host: str = "localhost"
    policy_port: int = 8000

    @property
    def safetensors_path(self) -> Path:
        """Retrieve the path to the ManiFlow policy's safetensors file."""
        return self.checkpoint_dir / self.safetensors_relative_path

    def validate(self) -> None:
        """Validate that the initialized ManiFlow configuration is valid (raises if not)."""
        if not self.checkpoint_dir.is_dir():
            raise RuntimeError(f"Checkpoint directory not found: {self.checkpoint_dir.resolve()}")

        if not self.safetensors_path.exists():
            raise RuntimeError(f"Safetensors file not found: {self.safetensors_path.resolve()}")


CHECKPOINT_ROOT = Path(os.environ.get("MANIFLOW_CHECKPOINT_ROOT", "checkpoints"))
"""Directory holding the ManiFlow checkpoint directories named below: `$MANIFLOW_CHECKPOINT_ROOT`,
or `checkpoints/` under the working directory."""

DATASET_ROOT = Path(os.environ.get("TOYBLOCKS_DATASET_ROOT", "datasets"))
"""Directory holding the LeRobot datasets the checkpoints were trained on, named below as
provenance only: `$TOYBLOCKS_DATASET_ROOT`, or `datasets/` under the working directory."""

MANIFLOW_CONFIGS: dict[str, ManiFlowConfig] = {
    "DD": ManiFlowConfig(
        checkpoint_dir=CHECKPOINT_ROOT
        / Path(
            ""
            "Ep37000-20000-5000_rgb224_sep3_reorient_PARTIAL_0910_1618_sep12_pnp_play_"
            "render_N1_sep10_pnp_ai2led_allrand_N1_50-09-12-23-55-43/"
        ),
        safetensors_relative_path="checkpoints/steps_200000_model.safetensors",
        dataset_dirs=(DATASET_ROOT / Path("sep3_reorient_PARTIAL_0910_1618"),),
        observation_mode="state+rgb",
    ),  #  "SAB RGB policy"
    "DPH": ManiFlowConfig(
        checkpoint_dir=CHECKPOINT_ROOT
        / Path(
            "DPH-simdepth-mix-Ep37000-10000-5000-"
            "11000_rgb224_sep3_reorient_PARTIAL_0910_1618_sep12_pnp_play_render_N1_sep10_p-09-24-12-23-32"
        ),
        safetensors_relative_path="checkpoints/steps_200000_model.safetensors",
        dataset_dirs=(
            DATASET_ROOT / Path("sep3_reorient_PARTIAL_0910_1618"),
            DATASET_ROOT / Path("sep12_pnp_play_render_N1"),
            DATASET_ROOT / Path("sep10_pnp_ai2led_allrand_N1"),
            DATASET_ROOT / Path("sep12_pnp_play_random_pick_render_N1"),
        ),
        observation_mode="state+rgb+depth",
    ),  # "RGBD policy with ideal depth"
    "LE": ManiFlowConfig(
        checkpoint_dir=CHECKPOINT_ROOT
        / Path(
            "rgb224_oct06_stack3_1000x32_50_25_abs_"
            "token_shared_rgb_mask0.1_sigma0.8_lang_t5-small_action_only_pause-10-15-18-53-01"
        ),
        safetensors_relative_path="checkpoints/steps_70000_model.safetensors",
        dataset_dirs=(DATASET_ROOT / Path("oct06_stack3_1000x32"),),
        observation_mode="state+rgb",
    ),
    "LEA": ManiFlowConfig(
        checkpoint_dir=CHECKPOINT_ROOT
        / Path(
            "rgb224_oct06_stack3_1000x32_50_25_"
            "abs_token_shared_rgb_mask0.1_sigma0.8_lang_t5-small_action_only_pause-10-17-10-28-48"
        ),
        safetensors_relative_path="checkpoints/steps_200000_model.safetensors",
        dataset_dirs=(DATASET_ROOT / Path("oct06_stack3_1000x32"),),
        observation_mode="state+rgb",
    ),
    "LO": ManiFlowConfig(
        checkpoint_dir=CHECKPOINT_ROOT
        / Path(
            "oct09_stack2_rg855_1000x32_"
            "oct09_stack3_rg855_1000x32_oct06_stack3_glyphcolor_N1_oct12_stack4_rg855_1450x32_"
            "oct21-10-22-19-30-47"
        ),
        safetensors_relative_path="checkpoints/steps_150000_model.safetensors",
        dataset_dirs=(DATASET_ROOT / Path("oct12_stack4_rg855_1450x32"),),
        observation_mode="state+rgb",
    ),
}

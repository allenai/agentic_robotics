"""PD-gain profiles for the Franka arm joints, by name.

The RL-trained rotate and flip skills were trained under one of these profiles, and the scripted
skills switch the arm to it for the rollout and back afterwards. Carried over from the TAMP
library's ``tmp3.Backends._shared.stiffness_profiles``, which is where the values were tuned.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class StiffnessProfile:
    """Immutable PD-gain profile for the seven arm joints."""

    stiffness: tuple[float, ...]
    damping: tuple[float, ...]
    effort_limit: tuple[float, ...]
    velocity_limit: tuple[float, ...]
    settle_steps: int = 10
    pd_mode: float = 1.0
    """Scalar written into recorded data: 0.0 is the softest profile, 1.0 the stiffest."""


STIFFNESS_PROFILES: dict[str, StiffnessProfile] = {
    "stiff": StiffnessProfile(
        stiffness=(600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0),
        damping=(50.0, 50.0, 50.0, 50.0, 30.0, 25.0, 15.0),
        effort_limit=(87.0, 87.0, 87.0, 87.0, 87.0, 87.0, 87.0),
        velocity_limit=(2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61),
        settle_steps=12,
        pd_mode=1.0,
    ),
    # Half of "stiff" on the main joints, damping scaled about critically (kd ~ sqrt(kp)), wrist
    # effort limits at the real FR3's 12 Nm.
    "medium": StiffnessProfile(
        stiffness=(300.0, 300.0, 300.0, 300.0, 150.0, 100.0, 50.0),
        damping=(35.0, 35.0, 35.0, 35.0, 20.0, 15.0, 10.0),
        effort_limit=(87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0),
        velocity_limit=(2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61),
        settle_steps=12,
        pd_mode=1.0,
    ),
    "soft": StiffnessProfile(
        stiffness=(100.0, 100.0, 100.0, 100.0, 75.0, 150.0, 50.0),
        damping=(20.0, 20.0, 20.0, 20.0, 7.5, 15.0, 5.0),
        effort_limit=(87.0, 87.0, 87.0, 87.0, 87.0, 87.0, 87.0),
        velocity_limit=(2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61),
        settle_steps=12,
        pd_mode=0.0,
    ),
    # Matches the ``fr3_robotiq_gripper_softer`` robot config's arm PD gains.
    "softer": StiffnessProfile(
        stiffness=(40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0),
        damping=(13.0, 13.0, 13.0, 13.0, 13.0, 13.0, 13.0),
        effort_limit=(87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0),
        velocity_limit=(2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61),
        settle_steps=12,
        pd_mode=0.0,
    ),
    "flip": StiffnessProfile(
        stiffness=(200.0, 200.0, 200.0, 200.0, 50.0, 50.0, 30.0),
        damping=(40.0, 40.0, 40.0, 40.0, 10.0, 10.0, 8.0),
        effort_limit=(87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0),
        velocity_limit=(2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61),
        settle_steps=12,
        pd_mode=1.0,
    ),
}

ROBOT_ID_TO_PROFILE: dict[str, str] = {
    "franka_robotiq_gripper": "soft",
    "franka_robotiq_gripper_soft": "soft",
    "fr3_robotiq_gripper_soft": "soft",
    "fr3_robotiq_gripper_softer": "softer",
}
"""The profile that reproduces each ToyBlocks robot config's arm PD gains, by the config's id."""


def get_profile(name: str) -> StiffnessProfile:
    """The named profile.

    :raises ValueError: Naming the known profiles, if there is none by that name
    """
    if name not in STIFFNESS_PROFILES:
        raise ValueError(
            f"Unknown stiffness profile {name!r}. Available: {sorted(STIFFNESS_PROFILES)}"
        )
    return STIFFNESS_PROFILES[name]


def get_profile_tensors(
    name: str, device: torch.device, batch_size: int = 1
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The named profile's stiffness, damping, effort limit and velocity limit, each shaped
    ``(batch_size, 7)``."""
    p = get_profile(name)
    stiffness = torch.tensor([p.stiffness], device=device).expand(batch_size, -1)
    damping = torch.tensor([p.damping], device=device).expand(batch_size, -1)
    effort = torch.tensor([p.effort_limit], device=device).expand(batch_size, -1)
    velocity_limit = torch.tensor([p.velocity_limit], device=device).expand(batch_size, -1)
    return stiffness, damping, effort, velocity_limit


def profile_for_robot_id(robot_id: str | None, default: str = "soft") -> str:
    """The profile matching a ToyBlocks robot config id, or ``default`` for an unknown one."""
    if robot_id is None:
        return default
    return ROBOT_ID_TO_PROFILE.get(robot_id, default)

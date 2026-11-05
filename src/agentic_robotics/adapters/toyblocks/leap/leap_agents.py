"""Define abstract base classes for all method and baseline agents used in the LEAP paper."""

from .context_clearing_agent import ContextClearingAgent
from .filesystem_tools_agent import FilesystemAgent
from .motor_skills_agent import MotorSkillsAgent
from .perception_tools_agent import PerceptionToolsAgent
from .recursive_agent import RecursiveAgent
from .scientific_agent import ScientificAgent
from .simulator_tools_agent import SimulatorToolsAgent
from .update_docstring_agent import UpdateDocstringAgent


class EvaluationAgent(MotorSkillsAgent, PerceptionToolsAgent):
    """An agent defining the toolset (but not docstrings) provided to all approaches during the
    evaluation phase.

    Note that each agent should use the appropriate docstrings for the motor skills:
        - Control: Original docstrings
        - Oracle: Human-designed 'ground-truth' docstrings
        - Naive: Docstrings (presumably) adapted during experimentation
        - Harness: Docstrings (presumably) adapted during experimentation
    """


class ControlAgent(EvaluationAgent):
    """A control agent, with original-docstring motor skills and perception tools."""


# Docstrings for the Oracle agent are defined in ORACLE_DOCSTRINGS in oracle_agent.py.


class NaiveAgent(
    MotorSkillsAgent,
    PerceptionToolsAgent,
    FilesystemAgent,
    SimulatorToolsAgent,
    UpdateDocstringAgent,
    RecursiveAgent,
    ContextClearingAgent,
):
    """A naive, experiment-capable agent able to update its docstrings, spawn subagents, etc."""


class NoOpMotorSkillsAgent(MotorSkillsAgent):
    """An agent with the standard motor skills interface, but whose motor skills cannot execute."""

    @staticmethod
    def get_refusal_text(tool_name: str) -> str:
        """Construct the refusal message given to the agent when it tries to call a motor skill."""
        return f"Cannot call '{tool_name}'. You cannot execute your motor skills directly."


class HarnessAgent(
    NoOpMotorSkillsAgent,
    PerceptionToolsAgent,
    FilesystemAgent,
    SimulatorToolsAgent,
    UpdateDocstringAgent,
    ScientificAgent,
    ContextClearingAgent,
):
    """An experiment-capable agent with tools forcing it to follow the scientific method."""

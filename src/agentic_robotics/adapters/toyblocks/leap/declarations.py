"""Register local tools against the LEAP ABC declarations, and check a served tool set against
a baseline's declaration before the first model call.

The ABCs under `src/agentic_robotics/adapters/toyblocks/leap/` are the model-facing interface:
names, parameter schemas and docstrings. A skill server inherits the ABC whose tools it
implements, and `ToyBlocksMCPServer._register_tool` checks each skill against it. The local tools
(files, docstring updates, context clearing, subagents, experiments) are plain functions closed
over run state, so they cannot inherit; `register_declared_tool` is their equivalent, and the
same `assert_matches_declaration` does the comparing.

`assert_serves` is the boot-time check: the tools an agent is about to be handed must be exactly
the baseline's abstract methods, with the declared description (or the description set given,
for an evaluation run on learned docstrings) and the schema the SDK derives from the declared
signature. It runs on the listing the agent will see, so nothing registered on any source can
slip past it.
"""

from __future__ import annotations

from inspect import cleandoc
from typing import TYPE_CHECKING, Any

from mcp.server.mcpserver.utilities.func_metadata import func_metadata

from agentic_robotics.adapters.toyblocks.toyblocks_mcp_server import assert_matches_declaration

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

    from mcp.server.mcpserver import MCPServer
    from mcp_types import Tool as MCPTool


def declaring_member(agent_cls: type, name: str) -> Callable:
    """The furthest ancestor's definition of `name`, the way the skill servers resolve it.

    :raises AttributeError: If no class in the MRO defines `name`
    """
    for klass in reversed(agent_cls.__mro__):
        member = klass.__dict__.get(name)
        if member is not None:
            return member
    raise AttributeError(f"{agent_cls.__name__} declares no tool named {name!r}.")


def declared_description(declaration: Callable) -> str:
    """The model-facing description of a declared tool: its docstring, dedented."""
    if not declaration.__doc__:
        raise ValueError(f"Declaration {declaration.__qualname__} has no docstring.")
    return cleandoc(declaration.__doc__)


def declared_schema(declaration: Callable) -> dict[str, Any]:
    """The input schema the SDK derives from the declared signature, `self` skipped."""
    return func_metadata(declaration, skip_names=["self"]).arg_model.model_json_schema()


def register_declared_tool(
    server: MCPServer, agent_cls: type, name: str, implementation: Callable
) -> str:
    """Register `implementation` as the tool `agent_cls` declares under `name`.

    The served description is the declaration's docstring, never the implementation's, and the
    implementation's signature must match the declaration parameter by parameter.

    :return: The tool's name, for allowlists
    :raises TypeError: If the implementation's signature differs from the declaration
    """
    declaration = declaring_member(agent_cls, name)
    assert_matches_declaration(implementation, declaration, owner=implementation.__module__)
    server.tool(name=name, description=declared_description(declaration))(implementation)
    return name


def declared_tools(agent_cls: type) -> dict[str, tuple[str, dict[str, Any]]]:
    """Every tool a baseline declares: name to (description, input schema)."""
    tools = {}
    for name in sorted(agent_cls.__abstractmethods__):
        declaration = declaring_member(agent_cls, name)
        tools[name] = (declared_description(declaration), declared_schema(declaration))
    return tools


def declared_descriptions(agent_cls: type) -> dict[str, str]:
    """Every tool a baseline declares, name to its declared description.

    What the composed server installs on entry when no other set is given: the declared
    docstrings are what Control is evaluated with and what Naive and Harness start from,
    whichever skill module serves the tools, so the served text need not equal the declaration
    on its own. The schema must.
    """
    return {name: text for name, (text, _) in declared_tools(agent_cls).items()}


def assert_serves(
    tools: Iterable[MCPTool],
    agent_cls: type,
    descriptions: Mapping[str, str] | None = None,
) -> None:
    """Refuse to start unless `tools` is exactly what `agent_cls` declares.

    :param tools: The listing the agent will see
    :param agent_cls: The baseline (Control, Naive, Harness, ...) for this run and phase
    :param descriptions: The description each tool must carry instead of the declared one, for
        an evaluation run on a learned docstring set; every served tool must be present in it
    :raises RuntimeError: Naming every tool that is missing, extra, or differs in description
        or schema
    """
    served = {tool.name: tool for tool in tools}
    expected = declared_tools(agent_cls)
    problems = []

    missing = sorted(set(expected) - set(served))
    extra = sorted(set(served) - set(expected))
    if missing:
        problems.append(f"missing tools: {missing}")
    if extra:
        problems.append(f"tools not declared by {agent_cls.__name__}: {extra}")

    for name in sorted(set(expected) & set(served)):
        declared_text, schema = expected[name]
        if descriptions is not None:
            if name not in descriptions:
                problems.append(f"{name}: no description in the given set")
                continue
            declared_text = descriptions[name]
        tool = served[name]
        if (tool.description or "") != declared_text:
            problems.append(
                f"{name}: description differs\n  expected: {declared_text!r}\n"
                f"  served:   {tool.description!r}"
            )
        if tool.input_schema != schema:
            problems.append(
                f"{name}: input schema differs\n  expected: {schema}\n"
                f"  served:   {tool.input_schema}"
            )

    if problems:
        raise RuntimeError(
            f"The served tools do not match {agent_cls.__name__}'s declaration:\n- "
            + "\n- ".join(problems)
        )


def recompose_on_backend(agent_cls: type, backend: str) -> type:
    """`agent_cls` with its motor-skill base swapped for the one `backend` serves.

    The baselines in `leap_agents.py` are composed from the scripted motor-skill ABC
    (`motor_skills_agent.MotorSkillsAgent`). The ManiFlow skills carry a `steps` parameter and
    their own docstrings, declared in `maniflow_skills/maniflow_motor_skills_agent.py`, so under
    the ManiFlow backend the same composition is rebuilt on that ABC; the composition itself stays
    single-sourced in `leap_agents.py`, only the motor-skill base is swapped. Any other backend
    returns `agent_cls` unchanged.

    :param agent_cls: A baseline composed from the LEAP ABCs
    :param backend: `config.skill_backend` as a string
    """
    if str(backend) != "maniflow":
        return agent_cls
    from agentic_robotics.adapters.toyblocks.leap.leap_agents import NoOpMotorSkillsAgent
    from agentic_robotics.adapters.toyblocks.leap.maniflow_skills import (
        maniflow_motor_skills_agent as maniflow_declarations,
    )
    from agentic_robotics.adapters.toyblocks.leap.motor_skills_agent import MotorSkillsAgent

    ManiFlowMotorSkillsAgent = maniflow_declarations.MotorSkillsAgent
    no_op = type(
        NoOpMotorSkillsAgent.__name__,
        (ManiFlowMotorSkillsAgent,),
        {"get_refusal_text": NoOpMotorSkillsAgent.__dict__["get_refusal_text"]},
    )
    swap = {MotorSkillsAgent: ManiFlowMotorSkillsAgent, NoOpMotorSkillsAgent: no_op}
    if not any(base in swap for base in agent_cls.__bases__):
        return agent_cls
    return type(
        agent_cls.__name__,
        tuple(swap.get(base, base) for base in agent_cls.__bases__),
        {"__doc__": agent_cls.__doc__},
    )

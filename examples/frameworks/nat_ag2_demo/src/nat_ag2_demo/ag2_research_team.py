"""AG2 (formerly AutoGen) research team demo for NAT.

Two agents (researcher + writer) collaborate via GroupChat
to produce a research summary with NAT profiling integration.
"""

from pydantic import BaseModel, Field

from autogen import ConversableAgent
from autogen.agentchat import initiate_group_chat
from autogen.agentchat.group.patterns import AutoPattern

from nat.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig


class AG2ResearchConfig(FunctionBaseConfig):
    """Configuration for the AG2 research team."""

    task: str = Field(
        default="Research and summarize recent advances "
        "in AI agent frameworks.",
        description="The research topic.",
    )
    llm_config: str = Field(
        default="default",
        description="NAT LLM config name.",
    )
    max_rounds: int = Field(
        default=10,
        description="Max GroupChat rounds.",
    )


@register_function(
    config_type=AG2ResearchConfig,
    framework_wrappers=[LLMFrameworkEnum.AG2],
)
async def ag2_research_team(
    config: AG2ResearchConfig,
    builder: Builder,
) -> str:
    """Run a 2-agent research team with AG2."""
    llm_config = await builder.get_llm(
        config.llm_config,
        wrapper_type=LLMFrameworkEnum.AG2,
    )

    researcher = ConversableAgent(
        name="researcher",
        system_message=(
            "You are a research specialist. Investigate the "
            "topic thoroughly. Present key facts, data, and "
            "sources in a structured format."
        ),
        llm_config=llm_config,
    )

    writer = ConversableAgent(
        name="writer",
        system_message=(
            "You are a technical writer. Synthesize the "
            "researcher's findings into a clear, structured "
            "summary with: Key Findings, Analysis, and "
            "Recommendations. Keep it under 500 words."
        ),
        llm_config=llm_config,
    )

    user = ConversableAgent(
        name="user", human_input_mode="NEVER"
    )

    pattern = AutoPattern(
        initial_agent=researcher,
        agents=[researcher, writer],
        user_agent=user,
        group_manager_args={"llm_config": llm_config},
    )

    result, ctx, last = initiate_group_chat(
        pattern=pattern,
        messages=config.task,
        max_rounds=config.max_rounds,
    )

    # Return last substantive message
    for msg in reversed(result.chat_history):
        content = msg.get("content", "")
        if content and "TERMINATE" not in content:
            return content

    return "Research complete."

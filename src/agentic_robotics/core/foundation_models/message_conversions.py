"""Define utilities to convert between message representations for foundation models."""

from typing import Literal

from openai.types.responses import EasyInputMessageParam, ResponseInputTextParam

ConversationRole = Literal["user", "assistant", "developer", "system"]


def create_openai_message(
    text: str, role: ConversationRole = "user", caching: bool = False
) -> EasyInputMessageParam:
    """Create an OpenAI-API-compatible message from the given text.

    :param text: Text content in the message
    :param role: Conversation role used for the message (default: "user")
    :param caching: If True, adds a cache breakpoint to the message (default: False)
    :return: Constructed EasyInputMessageParam message
    """
    content_block = ResponseInputTextParam(text=text, type="input_text")
    if caching:
        content_block["prompt_cache_breakpoint"] = {"mode": "explicit"}

    return EasyInputMessageParam(content=[content_block], role=role)

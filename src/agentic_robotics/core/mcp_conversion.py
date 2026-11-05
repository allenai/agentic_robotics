"""Define functions for converting between the MCP Python SDK and the OpenAI API."""

from mcp_types import CallToolResult, ImageContent, TextContent
from mcp_types import Tool as MCPTool
from openai.types.responses import (
    FunctionToolParam,
    ResponseFunctionCallOutputItemListParam,
    ResponseInputImageContentParam,
    ResponseInputTextContentParam,
)


def mcp_tool_to_openai_tool(tool: MCPTool) -> FunctionToolParam:
    """Convert an MCP tool definition into the format expected by the OpenAI Responses API."""
    return FunctionToolParam(
        name=tool.name,
        parameters=tool.input_schema,
        type="function",
        description=tool.description,
        strict=False,
    )


def mcp_result_to_openai_output(result: CallToolResult) -> ResponseFunctionCallOutputItemListParam:
    """Convert the result of an MCP tool call into the output of an OpenAI function call."""
    output: ResponseFunctionCallOutputItemListParam = []

    if result.is_error:
        output.append(
            ResponseInputTextContentParam(
                text="The tool call failed. The error is described below.", type="input_text"
            )
        )

    for block in result.content:
        match block:
            case TextContent():
                output.append(ResponseInputTextContentParam(text=block.text, type="input_text"))

            case ImageContent():
                uri = f"data:{block.mime_type};base64,{block.data}"
                output.append(ResponseInputImageContentParam(image_url=uri, type="input_image"))

            case _:
                raise NotImplementedError(f"Unrecognized content type: {type(block)}.")

    return output

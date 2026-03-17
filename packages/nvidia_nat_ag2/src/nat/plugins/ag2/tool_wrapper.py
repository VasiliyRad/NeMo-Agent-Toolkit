"""Converts NAT Function objects to AG2 Tool objects."""

import asyncio
import concurrent.futures
from typing import Any

from autogen.tools import Tool

from nat.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.cli.register_workflow import register_tool_wrapper
from nat.builder.function import Function


_tool_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=4
)


@register_tool_wrapper(wrapper_type=LLMFrameworkEnum.AG2)
def ag2_tool_wrapper(
    name: str,
    fn: Function,
    _builder: Builder,
) -> Tool:
    """Convert a NAT Function to an AG2 Tool."""

    def call_function(**kwargs: Any) -> Any:
        """Execute the NAT function synchronously."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(fn.acall(**kwargs))
        future = _tool_executor.submit(
            asyncio.run, fn.acall(**kwargs)
        )
        return future.result()

    call_function.__name__ = name
    call_function.__doc__ = fn.description or name

    return Tool(
        name=name,
        description=fn.description or name,
        func_or_tool=call_function,
        parameters_json_schema=(
            fn.input_schema.model_json_schema()
            if fn.input_schema
            else None
        ),
    )

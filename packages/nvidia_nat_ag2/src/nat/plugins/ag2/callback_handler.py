# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Profiler callback handler for AG2.

Patches AG2's OpenAIWrapper to capture LLM call timings and
token usage for NAT's profiling pipeline.
"""

import threading
import time
from typing import Any

from nat.builder.context import Context
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.data_models.intermediate_step import IntermediateStepPayload
from nat.data_models.intermediate_step import IntermediateStepType
from nat.data_models.intermediate_step import StreamEventData
from nat.data_models.intermediate_step import UsageInfo
from nat.data_models.profiler_callback import BaseProfilerCallback
from nat.data_models.token_usage import TokenUsageBaseModel


class AG2ProfilerHandler(BaseProfilerCallback):
    """Instruments AG2 agents for NAT profiling."""

    _original_create = None
    _patch_lock = threading.Lock()
    _patch_count = 0

    def __init__(self) -> None:
        """Initialize the AG2ProfilerHandler."""
        super().__init__()
        self._lock = threading.Lock()
        self.last_call_ts = time.time()
        self.step_manager = Context.get().intermediate_step_manager

    def patch(self) -> None:
        """Monkey-patch AG2's LLM calling to emit profiling
        events."""
        try:
            from autogen.oai.client import OpenAIWrapper
        except ImportError:
            return

        with AG2ProfilerHandler._patch_lock:
            AG2ProfilerHandler._patch_count += 1
            if AG2ProfilerHandler._original_create is not None:
                return  # Already patched

            AG2ProfilerHandler._original_create = (
                OpenAIWrapper.create
            )

        # Use module-level list so multiple handlers can
        # receive events
        handler = self

        def patched_create(wrapper_self: Any, *args: Any, **kwargs: Any) -> Any:
            now = time.time()
            with handler._lock:
                seconds_between_calls = int(now - handler.last_call_ts)

            model_name = kwargs.get("model", "")

            start_payload = IntermediateStepPayload(
                event_type=IntermediateStepType.LLM_START,
                framework=LLMFrameworkEnum.AG2,
                name=model_name,
                data=StreamEventData(input=str(kwargs)),
                usage_info=UsageInfo(
                    token_usage=TokenUsageBaseModel(),
                    num_llm_calls=1,
                    seconds_between_calls=seconds_between_calls,
                ),
            )
            start_uuid = start_payload.UUID
            handler.step_manager.push_intermediate_step(start_payload)

            try:
                result = (
                    AG2ProfilerHandler._original_create(
                        wrapper_self, *args, **kwargs
                    )
                )
                end_time = time.time()
                handler.step_manager.push_intermediate_step(
                    IntermediateStepPayload(
                        event_type=IntermediateStepType.LLM_END,
                        span_event_timestamp=end_time,
                        framework=LLMFrameworkEnum.AG2,
                        name=model_name,
                        data=StreamEventData(
                            input=str(kwargs),
                            output=str(result),
                        ),
                        usage_info=UsageInfo(
                            token_usage=TokenUsageBaseModel(),
                            num_llm_calls=1,
                            seconds_between_calls=seconds_between_calls,
                        ),
                        UUID=start_uuid,
                    )
                )
                with handler._lock:
                    handler.last_call_ts = end_time
                return result
            except Exception as e:
                handler.step_manager.push_intermediate_step(
                    IntermediateStepPayload(
                        event_type=IntermediateStepType.LLM_END,
                        span_event_timestamp=time.time(),
                        framework=LLMFrameworkEnum.AG2,
                        name=model_name,
                        data=StreamEventData(
                            input=str(kwargs),
                            output=str(e),
                        ),
                        usage_info=UsageInfo(
                            token_usage=TokenUsageBaseModel(),
                        ),
                        UUID=start_uuid,
                    )
                )
                with handler._lock:
                    handler.last_call_ts = time.time()
                raise

        OpenAIWrapper.create = patched_create

    def unpatch(self) -> None:
        """Restore original AG2 methods."""
        with AG2ProfilerHandler._patch_lock:
            if AG2ProfilerHandler._patch_count <= 0:
                return
            AG2ProfilerHandler._patch_count -= 1
            if AG2ProfilerHandler._patch_count > 0:
                return  # Other handlers still active
            if AG2ProfilerHandler._original_create is None:
                return
            try:
                from autogen.oai.client import OpenAIWrapper

                OpenAIWrapper.create = (
                    AG2ProfilerHandler._original_create
                )
                AG2ProfilerHandler._original_create = None
            except ImportError:
                pass

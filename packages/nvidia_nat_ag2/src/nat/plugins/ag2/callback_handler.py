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
token usage for the profiling pipeline.
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
    """Instruments AG2 agents for profiling."""

    _original_create = None
    _patch_lock = threading.Lock()
    _handlers: list["AG2ProfilerHandler"] = []

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
            AG2ProfilerHandler._handlers.append(self)
            if AG2ProfilerHandler._original_create is not None:
                return  # Already patched, just registered

            AG2ProfilerHandler._original_create = (
                OpenAIWrapper.create
            )

        def patched_create(wrapper_self: Any, *args: Any, **kwargs: Any) -> Any:
            # Capture locally to avoid race with unpatch()
            original = AG2ProfilerHandler._original_create
            if original is None:
                return wrapper_self.create(*args, **kwargs)

            # Snapshot handlers to iterate
            with AG2ProfilerHandler._patch_lock:
                handlers = list(AG2ProfilerHandler._handlers)

            model_name = kwargs.get("model", "")
            sanitized_input = {
                "model": model_name,
                "message_count": len(kwargs.get("messages", [])),
            }

            # Emit LLM_START to all handlers
            start_uuids = {}
            for h in handlers:
                now = time.time()
                with h._lock:
                    seconds_between_calls = int(now - h.last_call_ts)
                payload = IntermediateStepPayload(
                    event_type=IntermediateStepType.LLM_START,
                    framework=LLMFrameworkEnum.AG2,
                    name=model_name,
                    data=StreamEventData(input=str(sanitized_input)),
                    usage_info=UsageInfo(
                        token_usage=TokenUsageBaseModel(),
                        num_llm_calls=1,
                        seconds_between_calls=seconds_between_calls,
                    ),
                )
                start_uuids[id(h)] = payload.UUID
                h.step_manager.push_intermediate_step(payload)

            try:
                result = original(wrapper_self, *args, **kwargs)
                end_time = time.time()
                for h in handlers:
                    h.step_manager.push_intermediate_step(
                        IntermediateStepPayload(
                            event_type=IntermediateStepType.LLM_END,
                            span_event_timestamp=end_time,
                            framework=LLMFrameworkEnum.AG2,
                            name=model_name,
                            data=StreamEventData(
                                input=str(sanitized_input),
                                output="completed",
                            ),
                            usage_info=UsageInfo(
                                token_usage=TokenUsageBaseModel(),
                                num_llm_calls=1,
                            ),
                            UUID=start_uuids.get(id(h)),
                        )
                    )
                    with h._lock:
                        h.last_call_ts = end_time
                return result
            except Exception as e:
                err_time = time.time()
                for h in handlers:
                    h.step_manager.push_intermediate_step(
                        IntermediateStepPayload(
                            event_type=IntermediateStepType.LLM_END,
                            span_event_timestamp=err_time,
                            framework=LLMFrameworkEnum.AG2,
                            name=model_name,
                            data=StreamEventData(
                                input=str(sanitized_input),
                                output=f"error: {type(e).__name__}",
                            ),
                            usage_info=UsageInfo(
                                token_usage=TokenUsageBaseModel(),
                            ),
                            UUID=start_uuids.get(id(h)),
                        )
                    )
                    with h._lock:
                        h.last_call_ts = err_time
                raise

        OpenAIWrapper.create = patched_create

    def unpatch(self) -> None:
        """Restore original AG2 methods."""
        with AG2ProfilerHandler._patch_lock:
            if self in AG2ProfilerHandler._handlers:
                AG2ProfilerHandler._handlers.remove(self)
            if AG2ProfilerHandler._handlers:
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

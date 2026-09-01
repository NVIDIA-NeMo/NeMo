# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""vLLM-Omni implementations of the two component contracts.

Both are thin: the engines are process-scoped and owned by ``OmniRuntime``,
while the request state lives in the per-stream ``OmniStreamingSession`` these
classes read off the decode state. Importing this package does not import
vLLM.
"""

from typing import Any


def require_session(state: Any):
    """Return the stream's ``OmniStreamingSession``, or say why there isn't one.

    ``omni_session`` is a declared field on ``StreamingDecodeState``, so this
    reads it directly: a missing attribute is a programming error worth an
    AttributeError, while ``None`` is the real case worth explaining.
    """
    session = state.omni_session
    if session is None:
        raise RuntimeError(
            "A vllm_omni component requires a per-stream OmniStreamingSession; "
            "make sure begin_stream(...) ran for this stream before the first frame."
        )
    return session

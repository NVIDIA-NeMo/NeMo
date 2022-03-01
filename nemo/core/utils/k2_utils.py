# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Tuple

__K2_MINIMUM_MAJOR_VERSION__ = 1
__K2_MINIMUM_MINOR_VERSION__ = 11

K2_INSTALLATION_MESSAGE = (
    "Could not import `k2`.\n"
    "Please install k2 in one of the following ways:\n"
    "1) Run `bash scripts/speech_recognition/k2/setup.sh`\n"
    "2) (not recommended) Use any approach from https://k2-fsa.github.io/k2/installation/index.html "
    "if your your cuda and pytorch versions are supported.\n"
    "It is advised to always install k2 using setup.sh only, "
    "as different versions of k2 may not interact with the NeMo code as expected."
)
K2_IMPORT_SUCCESS = "Successfully imported k2."


def k2_is_available() -> Tuple[bool, str]:
    try:
        import k2

        return True, K2_IMPORT_SUCCESS
    except (ImportError, ModuleNotFoundError):
        pass
    return False, K2_INSTALLATION_MESSAGE


def k2_satisfies_minimum_version() -> Tuple[bool, str]:
    import k2

    k2_major_version, k2_minor_version = k2.__dev_version__.split(".")[:2]
    ver_satisfies = (
        int(k2_major_version) >= __K2_MINIMUM_MAJOR_VERSION__ and int(k2_minor_version) >= __K2_MINIMUM_MINOR_VERSION__
    )
    return (
        ver_satisfies,
        (
            f"Minimum required k2 version: {__K2_MINIMUM_MAJOR_VERSION__}.{__K2_MINIMUM_MINOR_VERSION__};\n"
            f"Installed k2 version: {k2_major_version}.{k2_minor_version}"
        ),
    )


def k2_import_guard():
    avail, msg = k2_is_available()
    if not avail:
        raise ModuleNotFoundError(msg)
    ver_ge, msg = k2_satisfies_minimum_version()
    if not ver_ge:
        raise ImportError(msg)

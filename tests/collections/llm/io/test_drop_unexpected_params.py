# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import fiddle as fdl

from nemo.lightning.io import drop_unexpected_params


class TestDropUnexpectedParams:

    def setup_method(self):
        """
        Setup common test resources.
        """

        class MockClassOld:
            def __init__(self, x, y, deprecated):
                pass

        class MockClassNew:
            def __init__(self, x, y):
                pass

        class OuterClass:
            def __init__(self, z, t):
                pass

        self.MockClassOld = MockClassOld
        self.MockClassNew = MockClassNew
        self.OuterClass = OuterClass

    def test_valid_config_stays_same(self):
        """
        Test that a valid config remains unchanged.
        """

        config = fdl.Config(self.MockClassNew, x=1, y=2)
        updated = drop_unexpected_params(config)

        assert not updated, "Expected the config to remain unchanged."
        assert config.x == 1
        assert config.y == 2

    def test_config_updates(self):
        """
        Test that a config with unexpected parameters gets updated.
        """
        config = fdl.Config(self.MockClassOld, x=1, y=2, deprecated=3)

        # Simulate deprecation issue by overriding target class
        config.__dict__['__fn_or_cls__'] = self.MockClassNew

        updated = drop_unexpected_params(config)
        assert updated, "Expected the config to be updated."
        assert config.x == 1
        assert config.y == 2
        assert not hasattr(config, "deprecated"), "Expected 'deprecated' to be removed from the config."

    def test_nested_config_updates(self):
        """
        Test that a nested config with unexpected parameters gets updated.
        """
        config = fdl.Config(self.OuterClass, z=4, t=fdl.Config(self.MockClassOld, x=1, y=2, deprecated=3))

        # Simulate deprecation issue by overriding target class
        config.t.__dict__["__fn_or_cls__"] = self.MockClassNew

        updated = drop_unexpected_params(config)
        assert updated, "Expected the nested config to be updated."
        assert config.z == 4
        assert config.t.x == 1
        assert config.t.y == 2
        assert not hasattr(config.t, "deprecated"), "Expected 'deprecated' to be removed from the inner config."

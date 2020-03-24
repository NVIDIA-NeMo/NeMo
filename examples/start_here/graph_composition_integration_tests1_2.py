# ! /usr/bin/python
# -*- coding: utf-8 -*-

# =============================================================================
# Copyright (c) 2020 NVIDIA. All Rights Reserved.
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
# =============================================================================

import nemo
from nemo.core import NeuralGraph, OperationMode, AppState

logging = nemo.logging

nf = nemo.core.NeuralModuleFactory()
# Instantiate the necessary neural modules.
dl = nemo.tutorials.RealFunctionDataLayer(n=10000, batch_size=128)
m2 = nemo.tutorials.TaylorNet(dim=4)
loss = nemo.tutorials.MSELoss()

logging.info("This example shows how one can build an `explicit` graph."
    F"It also shows how to activate and deactivate the g0 context `manually`")

# Create the g0 graph.
g0 = NeuralGraph(operation_mode=OperationMode.training)

# Activate the "g0 graph context" "manually" - all operations will be recorded to g0.
g0.activate()

# Define g0 - connections between the modules.
x, t = dl()
p = m2(x=x)
lss = loss(predictions=p, target=t)

# Deactivate the "g0 graph context" (this is really optional, as long as there are no other operations).
g0.deactivate()

# SimpleLossLoggerCallback will print loss values to console.
callback = nemo.core.SimpleLossLoggerCallback(
    tensors=[lss], print_func=lambda x: logging.info(f'Train Loss: {str(x[0].item())}'),
)

# Invoke "train" action.
nf.train([lss], callbacks=[callback], optimization_params={"num_epochs": 3, "lr": 0.0003}, optimizer="sgd")

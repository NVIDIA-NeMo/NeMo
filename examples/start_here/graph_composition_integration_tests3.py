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
from nemo.core import NeuralGraph, OperationMode

logging = nemo.logging

nf = nemo.core.NeuralModuleFactory()
# Instantiate the necessary neural modules.
dl = nemo.tutorials.RealFunctionDataLayer(n=10000, batch_size=128)
fx = nemo.tutorials.TaylorNet(dim=4)
loss = nemo.tutorials.MSELoss()

logging.info(
    "This example shows how one can nest one graph into another - with binding of the input ports."
    F" Please note that the nested graph can be used exatly like any other module"
    F" In particular, note that the input port 'x' of the module `m2` is bound in graph 'g2'"
    F" and then set to `x` returned by `dl` in the graph `g3`."
)

with NeuralGraph(operation_mode=OperationMode.training, name="g2") as g2:
    # Add module to graph and bind it input port 'x'.
    y = fx(x=g2)

# Build the training graph.
with NeuralGraph(operation_mode=OperationMode.training, name="g3") as g3:
    # Add modules to graph.
    x, t = dl()
    # Incorporate modules from existing graph.
    p = g2(x=x)
    lss = loss(predictions=p, target=t)

# SimpleLossLoggerCallback will print loss values to console.
callback = nemo.core.SimpleLossLoggerCallback(
    tensors=[lss], print_func=lambda x: logging.info(f'Train Loss: {str(x[0].item())}'),
)

# Invoke "train" action.
nf.train([lss], callbacks=[callback], optimization_params={"num_epochs": 3, "lr": 0.0003}, optimizer="sgd")

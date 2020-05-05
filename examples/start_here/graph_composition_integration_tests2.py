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

from nemo.backends.pytorch.tutorials import MSELoss, RealFunctionDataLayer, TaylorNet
from nemo.core import DeviceType, NeuralGraph, NeuralModuleFactory, OperationMode, SimpleLossLoggerCallback
from nemo.utils import logging

nf = NeuralModuleFactory(placement=DeviceType.CPU)
# Instantiate the necessary neural modules.
dl = RealFunctionDataLayer(n=100, batch_size=32, name="dl")
m2 = TaylorNet(dim=4, name="m2")
loss = MSELoss(name="loss")

logging.info(
    "This example shows how one can nest one graph into another - with binding of output ports."
    F" Please note that the nested graph can be used exatly like any other module"
    F" By default, all output graph ports are bound, thus `visible` outside."
)

with NeuralGraph(operation_mode=OperationMode.training, name="g1") as g1:
    x, t = dl()
    y = m2(x=x)

with NeuralGraph(operation_mode=OperationMode.training, name="g1.1") as g11:
    x1, t1, p1 = g1()
    lss = loss(predictions=p1, target=t1)

# SimpleLossLoggerCallback will print loss values to console.
callback = SimpleLossLoggerCallback(
    tensors=[lss], print_func=lambda x: logging.info(f'Train Loss: {str(x[0].item())}'),
)

# Invoke "train" action.
nf.train([lss], callbacks=[callback], optimization_params={"num_epochs": 2, "lr": 0.0003}, optimizer="sgd")

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

from collections.abc import MutableMapping

from nemo.utils import logging
from nemo.utils.connection import StepModulePort


class GraphOutput(object):
    """ A helper class represenging a single bound output. """

    def __init__(self, ntype, producer_step_module_port):
        """ 
        Initializes object.

        Args:
            type: a NeuralType object.
            producer_step_module_port: a producer StepModulePort tuple (step number (module name), port name).
        """
        self._ntype = ntype
        self._producer_step_module_port = producer_step_module_port

    @property
    def ntype(self):
        """ Returns NeuralType of that output. """
        return self._ntype

    @property
    def producer_step_module_port(self):
        """ Returns producer step port (step number (module), port name) tuple. """
        return self._producer_step_module_port


class GraphOutputs(MutableMapping):
    '''
        A specialized dictionary that contains bound outputs of a Neural Graph.
        In fact stores two lists of "outputs":
            - "default" outputs with default keys taken from outputs of modules (might result in
            overwriting some keys), and
            - "manual" used for specifying the subset of outputs, each with a new/different key
        When accessing the outputs, it returns the "manual" outputs. If "manual" outputs are not defined,
        will return/work on "default" outputs.
    '''

    def __init__(self, tensors_ref):
        """
            Initializes two (empty) dictionaries. 

            Args:
                tensors_ref - reference to neural graph's tensor (dict of dict).
        """

        # Tensors[step][output_port_name] passed from the external neural graph object.
        self._tensors_ref = tensors_ref

        # This dictionary stores the output tensors collected during the "default" tensor recording.
        # As they are using the default port names, the second/next tensor published on the same port
        # will overwrite the old one (Warning).
        self._default_outputs = {}

        # This dictionary stores list of output tensors of module "manually" indicated by the user.
        # In this case tring to overwriting the existing ports with new tensors will be forbidden (Exception).
        self._manual_outputs = {}

    def __setitem__(self, key, value):
        """
            This method is used to set the manual output - creates a GraphOutput item and adds it to the list.
            
            Args:
                key: name of the output (port).
                value: tensor that will be used to create GraphOutput.
        """
        # Make sure that user passed a NmTensor.
        assert type(value).__name__ == "NmTensor"
        if key in self._manual_outputs.keys():
            raise KeyError("Overwriting of a port `{}` that was previously manually bound is not allowed".format(key))
        # Ok, set output.
        self._manual_outputs[key] = GraphOutput(value.ntype, value.producer_step_module_port)

    def __getitem__(self, key):
        """ Returns GraphOutput - depending whether there are some manual outputs or not. """
        if len(self._manual_outputs) > 0:
            return self._manual_outputs[key]
        else:  # Use default dict.
            return self._default_outputs[key]

    def __delitem__(self, key):
        raise NotImplementedError("Deleting a bound output is not allowed")

    def __iter__(self):
        """ Iterates over the outputs - depending whether there are some manual outputs or not. """
        if len(self._manual_outputs) > 0:
            return iter(self._manual_outputs)
        else:  # Use default dict.
            return iter(self._default_outputs)

    def __len__(self):
        """ Return number of outputs - depending whether there are some manual outputs or not. """
        if len(self._manual_outputs) > 0:
            return len(self._manual_outputs)
        else:  # Use default dict.
            return len(self._default_outputs)

    def bind(self, tensors_ref, port_names=None):
        """ Binds the default outputs.

            Args:
                tensors_ref: List of tensors to be added.
                port_names: List of port names (visible outside). If None: using internal tensor "output port names".
        """
        # Set names.
        if port_names is None:
            port_names = [tensor.name for tensor in tensors_ref]

        for name, tensor in zip(port_names, tensors_ref):
            # Check the presence of the port name in "default" dictionary.
            if name in self._default_outputs.keys():
                # Name present - use the name being combination of producer and port names.
                name = (
                    str(tensor.producer_step_number) + "_" + tensor.producer_name + "_" + tensor.name
                )  # last = port name

                logging.warning(
                    "Setting unigue name of the default output port `{}` produced in step {} by `{}` to `{}`".format(
                        tensor.name, tensor.producer_step_number, tensor.producer_name, name
                    )
                )
            # Still, "overwrite" it.
            self._default_outputs[name] = GraphOutput(tensor.ntype, tensor.producer_step_module_port)

    @property
    def definitions(self):
        """ Property returns definitions of the output ports by extracting them on the fly from the bound outputs. """
        # Get the right output dictionary.
        d = self._manual_outputs if len(self._manual_outputs) > 0 else self._default_outputs

        # Extract port definitions (Neural Types).
        return {k: v.ntype for k, v in d.items()}

    @property
    def tensors(self):
        """
            Property returns output tensors by extracting them on the fly from the bound outputs.

            Returns:
                Dictionary of tensors in the format (output-name: tensor).
        """
        # Get the right output dictionary.
        d = self._manual_outputs if len(self._manual_outputs) > 0 else self._default_outputs

        output_tensors = {}
        # Get tensors by acessing the producer-ports.
        for k, v in d.items():
            producer_step = v.producer_step_module_port.step_number
            producer_port_name = v.producer_step_module_port.port_name
            # Find the right output tensor.
            tensor = self._tensors_ref[producer_step][producer_port_name]
            # Add it to the dictionary.
            output_tensors[k] = tensor
        # Return the result.
        return output_tensors

    @property
    def tensor_list(self):
        """
            Property returns output tensors by extracting them on the fly from the bound outputs.
            
            Returns:
                List of tensors.

        """
        # Get the right output dictionary.
        d = self._manual_outputs if len(self._manual_outputs) > 0 else self._default_outputs

        output_tensor_list = []
        # Get tensors by acessing the producer-ports.
        for k, v in d.items():
            producer_step = v.producer_step_module_port.step_number
            producer_port_name = v.producer_step_module_port.port_name
            # Find the right output tensor.
            tensor = self._tensors_ref[producer_step][producer_port_name]
            # Add it to the list.
            output_tensor_list.append(tensor)
        # Return the result.
        return output_tensor_list

    def serialize(self):
        """ Method responsible for serialization of the graph outputs.

            Returns:
                List containing mappings (step.module.output_port -> output | ntype).
        """
        serialized_outputs = {"mappings": []}

        # Get the right output dictionary.
        if len(self._manual_outputs) > 0:
            serialized_outputs["type"] = "manual"
            d = self._manual_outputs
        else:
            serialized_outputs["type"] = "default"
            d = self._default_outputs

        # Iterate through "bindings" (GraphOutputs).
        for key, binding in d.items():
            # Serialize: step.module.port -> output | ntype.
            smp = binding.producer_step_module_port
            source = str(smp.step_number) + "." + smp.module_name + "." + smp.port_name
            # Get type.
            ntype_str = str(binding.ntype)
            # Serialize!
            serialized_outputs["mappings"].append(source + "->" + key + " | " + ntype_str)
        # Return the result.
        return serialized_outputs

    def deserialize(self, serialized_outputs, modules):
        """ 
            Method responsible for deserialization of graph outputs.

            Args:
                serialized_outputs: A list of serialized outputs in the form of ("step.module.output_port->key | ntype")
                modules: List of modules required for neural type copying/checking.
        """
        # Check type.
        if serialized_outputs["type"] == "default":
            # We still need to deserialize.
            # Use-case: deserialization of a graph with nested graph with bound output.
            d = self._default_outputs
        else:
            d = self._manual_outputs

        # Iterate through serialized inputs one by one.
        for i in serialized_outputs["mappings"]:
            # Deserialize!
            [producer, key_ntype] = i.split("->")
            [key, ntype_str] = key_ntype.split(" | ")
            [step_number, producer_name, producer_port_name] = producer.split(".")
            # Get neural type from module output port definition.
            ntype = modules[producer_name].output_ports[producer_port_name]

            # Make sure the graph bound port type matches the deserialized type.
            assert ntype_str == str(ntype)

            # Create a new input.
            go = GraphOutput(ntype, StepModulePort(int(step_number), producer_name, producer_port_name))
            d[key] = go

        # Done.

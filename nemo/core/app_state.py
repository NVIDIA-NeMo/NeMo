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

import threading

# Sadly have to import this to avoid circular dependencies.
import nemo


class Singleton(type):
    """ Implementation of a generic singleton meta-class. """

    # List of instances - one per class.
    __instances = {}
    # Lock used for accessing the instance.
    __lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        """ Returns singleton instance.A thread safe implementation. """
        if cls not in cls.__instances:
            # Enter critical section.
            with cls.__lock:
                # Check once again.
                if cls not in cls.__instances:
                    # Create a new object instance - one per class.
                    cls.__instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        # Return the instance.
        return cls.__instances[cls]


class AppState(metaclass=Singleton):
    """
        Application state stores variables important from the point of view of execution of the NeMo application.
        Staring from the most elementary (epoch number, episode number, device used etc.) to the currently
        active graph etc.
    """

    def __init__(self, device=None):
        """
            Constructor. Initializes global variables.

            Args:
                device: main device used for computations [CPU | GPU] (DEFAULT: GPU)
        """
        # Had to set it to None in argument to avoid circular import at the class initialization phase.
        if device is None:
            self._device = nemo.core.DeviceType.GPU
        else:
            self._device = device
        self._neural_graph_manager = nemo.core.NeuralGraphManager()

    @property
    def graphs(self):
        """ Property returns the graph manager.

            Returns:
                List of created graphs
        """
        return self._neural_graph_manager

    def register_graph(self, graph):
        """ Registers a new graph. """
        self._neural_graph_manager.register_graph(graph)

    @property
    def active_graph(self):
        """ Property returns the active graph.

            Returns:
                Active graph
        """
        return self._neural_graph_manager.active_graph

    @active_graph.setter
    def active_graph(self, graph):
        """ Property sets the active graph.

            Args:
                graph: Neural graph object that will become active.
        """
        self._neural_graph_manager.active_graph = graph

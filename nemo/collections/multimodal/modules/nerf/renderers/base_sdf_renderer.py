# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
import torch

from nemo.collections.multimodal.modules.renderer.base_renderer import RendererBase


class BaseSDFRenderer(RendererBase):
    def __init__(self, bound):
        super().__init__(bound)

    # TODO(ahmadki): needs a rework
    @torch.no_grad()
    def get_vertices_and_triangles(self, resolution=None, S=128):
        deform = torch.tanh(self.deform) / self.grid_size

        vertices, triangles = self.dmtet(self.verts + deform, self.sdf, self.indices)

        vertices = vertices.detach().cpu().numpy()
        triangles = triangles.detach().cpu().numpy()

        return vertices, triangles

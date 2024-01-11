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
import torch.nn as nn


class NormalConsistencyLoss(nn.Module):
    def __init__(self):
        super(NormalConsistencyLoss, self).__init__()

    # TODO(ahmadki): is this safe to do in FP16 ?
    def forward(self, face_normals, t_pos_idx):
        tris_per_edge = self.compute_edge_to_face_mapping(t_pos_idx)

        # Fetch normals for both faces sharind an edge
        n0 = face_normals[tris_per_edge[:, 0], :]
        n1 = face_normals[tris_per_edge[:, 1], :]

        # Compute error metric based on normal difference
        term = torch.clamp(torch.sum(n0 * n1, -1, keepdim=True), min=-1.0, max=1.0)
        term = 1.0 - term

        return torch.mean(torch.abs(term))

    # TODO(ahmadki): should belog to mesh class
    def compute_edge_to_face_mapping(self, attr_idx):
        with torch.no_grad():
            # Get unique edges
            # Create all edges, packed by triangle
            all_edges = torch.cat(
                (
                    torch.stack((attr_idx[:, 0], attr_idx[:, 1]), dim=-1),
                    torch.stack((attr_idx[:, 1], attr_idx[:, 2]), dim=-1),
                    torch.stack((attr_idx[:, 2], attr_idx[:, 0]), dim=-1),
                ),
                dim=-1,
            ).view(-1, 2)

            # Swap edge order so min index is always first
            order = (all_edges[:, 0] > all_edges[:, 1]).long().unsqueeze(dim=1)
            sorted_edges = torch.cat(
                (torch.gather(all_edges, 1, order), torch.gather(all_edges, 1, 1 - order)), dim=-1
            )

            # Elliminate duplicates and return inverse mapping
            unique_edges, idx_map = torch.unique(sorted_edges, dim=0, return_inverse=True)

            tris = torch.arange(attr_idx.shape[0]).repeat_interleave(3).cuda()

            tris_per_edge = torch.zeros((unique_edges.shape[0], 2), dtype=torch.int64).cuda()

            # Compute edge to face table
            mask0 = order[:, 0] == 0
            mask1 = order[:, 0] == 1
            tris_per_edge[idx_map[mask0], 0] = tris[mask0]
            tris_per_edge[idx_map[mask1], 1] = tris[mask1]

            return tris_per_edge

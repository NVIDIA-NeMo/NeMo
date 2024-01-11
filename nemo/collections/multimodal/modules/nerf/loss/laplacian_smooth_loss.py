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


class LaplacianSmoothLoss(nn.Module):
    def __init__(self):
        super(LaplacianSmoothLoss, self).__init__()

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, verts, faces):
        with torch.no_grad():
            L = self.laplacian_uniform(verts, faces.long())
        loss = L.mm(verts)
        loss = loss.norm(dim=1)
        loss = loss.mean()
        return loss

    # TODO(ahmadki): should be moved to a separate mesh class
    def laplacian_uniform(self, verts, faces):
        V = verts.shape[0]
        F = faces.shape[0]

        # Neighbor indices
        ii = faces[:, [1, 2, 0]].flatten()
        jj = faces[:, [2, 0, 1]].flatten()
        adj = torch.stack([torch.cat([ii, jj]), torch.cat([jj, ii])], dim=0).unique(dim=1)
        adj_values = torch.ones(adj.shape[1], device=verts.device, dtype=torch.float)

        # Diagonal indices
        diag_idx = adj[0]

        # Build the sparse matrix
        idx = torch.cat((adj, torch.stack((diag_idx, diag_idx), dim=0)), dim=1)
        values = torch.cat((-adj_values, adj_values))

        # The coalesce operation sums the duplicate indices, resulting in the
        # correct diagonal
        return torch.sparse_coo_tensor(idx, values, (V, V)).coalesce()

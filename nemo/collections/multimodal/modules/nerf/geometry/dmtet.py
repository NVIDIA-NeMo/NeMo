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


class DeepMarchingTetrahedra:
    """
    Class for Deep Marching Tetrahedra (DMTet).

    Attributes:
        device (torch.device): Device to place the tensors.
        triangle_table (Tensor): Lookup table for the triangles.
        num_triangles_table (Tensor): Table for the number of triangles.
        base_tet_edges (Tensor): The base edges for the tetrahedrons.
    """

    def __init__(self, device: torch.device) -> None:
        """Initialize DMTet instance with the given device.

        Args:
            device (torch.device): The device to place the tensors on.
        """
        self.device = device
        self.triangle_table = self._create_triangle_table()
        self.num_triangles_table = self._create_num_triangles_table()
        self.base_tet_edges = self._create_base_tet_edges()

    def _create_triangle_table(self) -> torch.Tensor:
        """Create the lookup table for triangles.

        Returns:
            Tensor: The triangle lookup table.
        """
        return torch.tensor(
            [
                [-1, -1, -1, -1, -1, -1],
                [1, 0, 2, -1, -1, -1],
                [4, 0, 3, -1, -1, -1],
                [1, 4, 2, 1, 3, 4],
                [3, 1, 5, -1, -1, -1],
                [2, 3, 0, 2, 5, 3],
                [1, 4, 0, 1, 5, 4],
                [4, 2, 5, -1, -1, -1],
                [4, 5, 2, -1, -1, -1],
                [4, 1, 0, 4, 5, 1],
                [3, 2, 0, 3, 5, 2],
                [1, 3, 5, -1, -1, -1],
                [4, 1, 2, 4, 3, 1],
                [3, 0, 4, -1, -1, -1],
                [2, 0, 1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1],
            ],
            dtype=torch.long,
            device=self.device,
        )

    def _create_num_triangles_table(self) -> torch.Tensor:
        """Create the table for number of triangles.

        Returns:
            Tensor: The number of triangles table.
        """
        return torch.tensor([0, 1, 1, 2, 1, 2, 2, 1, 1, 2, 2, 1, 2, 1, 1, 0], dtype=torch.long, device=self.device)

    def _create_base_tet_edges(self) -> torch.Tensor:
        """Create the base edges for the tetrahedrons.

        Returns:
            Tensor: The base edges for tetrahedrons.
        """
        return torch.tensor([0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3], dtype=torch.long, device=self.device)

    def _sort_edges(self, edges_ex2: torch.Tensor) -> torch.Tensor:
        """Sort the given edges.

        Args:
            edges_ex2 (Tensor): The edges to be sorted.

        Returns:
            Tensor: The sorted edges.
        """
        with torch.no_grad():
            order = (edges_ex2[:, 0] > edges_ex2[:, 1]).long()
            order = order.unsqueeze(dim=1)
            a = torch.gather(input=edges_ex2, index=order, dim=1)
            b = torch.gather(input=edges_ex2, index=1 - order, dim=1)
        return torch.stack([a, b], -1)

    # TODO(ahmadki): rename to forward ? return mesh ?
    def __call__(self, positions: torch.Tensor, sdf_n: torch.Tensor, tet_fx4: torch.Tensor) -> tuple:
        """
        Process the provided data to generate vertices and faces.

        Args:
            positions (Tensor): Position tensor with shape [N, 3].
            sdf_n (Tensor): SDF tensor with shape [N].
            tet_fx4 (Tensor): Tetrahedron faces tensor with shape [F, 4].

        Returns:
            tuple: Vertices and faces tensors.
        """
        with torch.no_grad():
            occ_n = sdf_n > 0
            occ_fx4 = occ_n[tet_fx4.reshape(-1)].reshape(-1, 4)
            occ_sum = torch.sum(occ_fx4, -1)
            valid_tets = (occ_sum > 0) & (occ_sum < 4)
            occ_sum = occ_sum[valid_tets]

            # find all vertices
            all_edges = tet_fx4[valid_tets][:, self.base_tet_edges].reshape(-1, 2)
            all_edges = self._sort_edges(all_edges)
            unique_edges, idx_map = torch.unique(all_edges, dim=0, return_inverse=True)

            unique_edges = unique_edges.long()
            mask_edges = occ_n[unique_edges.reshape(-1)].reshape(-1, 2).sum(-1) == 1
            mapping = torch.ones((unique_edges.shape[0]), dtype=torch.long, device=self.device) * -1
            mapping[mask_edges] = torch.arange(mask_edges.sum(), dtype=torch.long, device=self.device)
            idx_map = mapping[idx_map]  # map edges to verts

            interp_v = unique_edges[mask_edges]

        edges_to_interp = positions[interp_v.reshape(-1)].reshape(-1, 2, 3)
        edges_to_interp_sdf = sdf_n[interp_v.reshape(-1)].reshape(-1, 2, 1)
        edges_to_interp_sdf[:, -1] *= -1

        denominator = edges_to_interp_sdf.sum(1, keepdim=True)
        edges_to_interp_sdf = torch.flip(edges_to_interp_sdf, [1]) / denominator
        verts = (edges_to_interp * edges_to_interp_sdf).sum(1)

        idx_map = idx_map.reshape(-1, 6)
        v_id = torch.pow(2, torch.arange(4, dtype=torch.long, device=self.device))
        tetindex = (occ_fx4[valid_tets] * v_id.unsqueeze(0)).sum(-1)
        num_triangles = self.num_triangles_table[tetindex]

        # Generate triangle indices
        faces = torch.cat(
            (
                torch.gather(
                    input=idx_map[num_triangles == 1],
                    dim=1,
                    index=self.triangle_table[tetindex[num_triangles == 1]][:, :3],
                ).reshape(-1, 3),
                torch.gather(
                    input=idx_map[num_triangles == 2],
                    dim=1,
                    index=self.triangle_table[tetindex[num_triangles == 2]][:, :6],
                ).reshape(-1, 3),
            ),
            dim=0,
        )

        return verts, faces

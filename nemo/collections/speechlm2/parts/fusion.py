# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""
Fusion modules for combining multi-modal embeddings in duplex speech models.

Supported fusion methods:
- "add" or None: Simple additive fusion with per-modality weights
- "concat": Concatenate (after LayerNorm) and project back to hidden_dim
- "gated_simple": Per-timestep, per-dimension gates with softmax normalization
- "gated_gmu": Gated Multimodal Unit (Arevalo et al., 2017) with per-modality sigmoid gates

For gated methods, the channel embeddings, tie_and_roll, and per-modality weights
are bypassed - the gating mechanism learns the optimal combination.
"""

import torch
import torch.nn as nn
from typing import Optional, Literal, Union

from nemo.utils import logging

FuseMethod = Literal["add", "concat", "gated_simple", "gated_gmu"]


class FusionModule(nn.Module):
    """Base class for fusion modules."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
    
    @property
    def uses_learned_gating(self) -> bool:
        """Return True if this fusion method uses learned gating (bypasses manual weights)."""
        return False
    
    def forward(
        self,
        agent_text_embeds: torch.Tensor,
        user_audio_embeds: torch.Tensor,
        user_text_embeds: Optional[torch.Tensor] = None,
        function_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Fuse multi-modal embeddings.
        
        Args:
            agent_text_embeds: Agent text embeddings (B, T, D)
            user_audio_embeds: User audio embeddings (B, T, D)
            user_text_embeds: User text/ASR embeddings (B, T, D) or None
            function_embeds: Function calling channel embeddings (B, T, D) or None
        
        Returns:
            Fused embeddings (B, T, D)
        """
        raise NotImplementedError


class AddFusion(FusionModule):
    """
    Simple additive fusion with optional per-modality weights.
    
    This is the default behavior matching the original implementation.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        agent_text_weight: float = 1.0,
        user_audio_weight: float = 1.0,
        user_text_weight: float = 1.0,
        function_weight: float = 1.0,
    ):
        super().__init__(hidden_dim)
        self.agent_text_weight = agent_text_weight
        self.user_audio_weight = user_audio_weight
        self.user_text_weight = user_text_weight
        self.function_weight = function_weight
    
    def forward(
        self,
        agent_text_embeds: torch.Tensor,
        user_audio_embeds: torch.Tensor,
        user_text_embeds: Optional[torch.Tensor] = None,
        function_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        output = agent_text_embeds * self.agent_text_weight
        output = output + user_audio_embeds * self.user_audio_weight
        if user_text_embeds is not None:
            output = output + user_text_embeds * self.user_text_weight
        if function_embeds is not None:
            output = output + function_embeds * self.function_weight
        return output


class ConcatFusion(FusionModule):
    """
    Concatenate embeddings (after LayerNorm) and project back to hidden_dim.
    
    Supports optional user_text_embeds and function_embeds; zeros are used when
    not provided. Uses 4 modalities (agent, audio, user_text, function) so that
    function channel can be optionally present.
    The linear layer learns to handle this case appropriately.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        agent_text_weight: float = 1.0,
        user_audio_weight: float = 1.0,
        user_text_weight: float = 1.0,
        function_weight: float = 1.0,
        num_modalities: int = 4,
    ):
        super().__init__(hidden_dim)
        self.agent_text_weight = agent_text_weight
        self.user_audio_weight = user_audio_weight
        self.user_text_weight = user_text_weight
        self.function_weight = function_weight
        self.num_modalities = num_modalities
        
        # LayerNorm for each modality
        self.ln_agent = nn.LayerNorm(hidden_dim)
        self.ln_audio = nn.LayerNorm(hidden_dim)
        self.ln_text = nn.LayerNorm(hidden_dim)
        self.ln_function = nn.LayerNorm(hidden_dim)
        
        # Projection layer (4 modalities; pad with zeros when optional channels missing)
        self.proj = nn.Linear(hidden_dim * num_modalities, hidden_dim, bias=False)
    
    def forward(
        self,
        agent_text_embeds: torch.Tensor,
        user_audio_embeds: torch.Tensor,
        user_text_embeds: Optional[torch.Tensor] = None,
        function_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Apply layernorm and weights
        agent_normed = self.ln_agent(agent_text_embeds) * self.agent_text_weight
        audio_normed = self.ln_audio(user_audio_embeds) * self.user_audio_weight
        
        if user_text_embeds is not None:
            text_normed = self.ln_text(user_text_embeds) * self.user_text_weight
        else:
            # Zero-pad when ASR embeddings are not available
            text_normed = torch.zeros_like(agent_text_embeds)
        
        if function_embeds is not None:
            function_normed = self.ln_function(function_embeds) * self.function_weight
        else:
            function_normed = torch.zeros_like(agent_text_embeds)
        
        # Concatenate along hidden dim and project
        concat = torch.cat([agent_normed, audio_normed, text_normed, function_normed], dim=-1)
        return self.proj(concat)


class GatedFusionSimple(FusionModule):
    """
    Learns a softmax-normalized gate over 4 streams at each timestep (agent, audio, user_text, function).
    
    Gate input: LayerNorm(each stream) → concatenate → linear → softmax
    Output: weighted sum of LayerNorm'd embeddings where weights are input-dependent.
    Optional channels (user_text, function) use zeros when not provided; their gate is masked to 0.
    
    Note: This is PER-TIMESTEP gating (one scalar weight per modality per timestep).
    For PER-DIMENSION gating, use GatedFusionGMU.
    """
    
    def __init__(self, hidden_dim: int, num_modalities: int = 4):
        super().__init__(hidden_dim)
        self.num_modalities = num_modalities
        
        # LayerNorm for each stream (normalizes distributions before gate computation)
        self.ln_agent = nn.LayerNorm(hidden_dim)
        self.ln_audio = nn.LayerNorm(hidden_dim)
        self.ln_text = nn.LayerNorm(hidden_dim)
        self.ln_function = nn.LayerNorm(hidden_dim)
        
        # Gate network: outputs one gate value per modality per timestep
        # Input: concatenated normalized modalities (B, T, D*4)
        # Output: gates (B, T, 4) - one scalar per modality
        self.gate_net = nn.Linear(
            hidden_dim * num_modalities,
            num_modalities,
            bias=True
        )
        
        # Initialize for uniform weighting
        self._init_uniform()
    
    @property
    def uses_learned_gating(self) -> bool:
        return True
    
    def _init_uniform(self):
        """Initialize gate network for uniform initial weighting."""
        nn.init.zeros_(self.gate_net.weight)
        nn.init.zeros_(self.gate_net.bias)
    
    def forward(
        self,
        agent_text_embeds: torch.Tensor,
        user_audio_embeds: torch.Tensor,
        user_text_embeds: Optional[torch.Tensor] = None,
        function_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, D = agent_text_embeds.shape
        
        has_user_text = user_text_embeds is not None
        has_function = function_embeds is not None
        text_input = user_text_embeds if has_user_text else torch.zeros_like(agent_text_embeds)
        function_input = function_embeds if has_function else torch.zeros_like(agent_text_embeds)
        
        # Apply LayerNorm to each stream before gate computation
        agent_normed = self.ln_agent(agent_text_embeds)
        audio_normed = self.ln_audio(user_audio_embeds)
        text_normed = self.ln_text(text_input)
        function_normed = self.ln_function(function_input)
        
        # Concatenate normalized inputs for gate computation
        concat = torch.cat([agent_normed, audio_normed, text_normed, function_normed], dim=-1)
        
        # Compute gates: (B, T, D*4) -> (B, T, 4)
        gates = self.gate_net(concat)  # (B, T, num_modalities)
        
        # Mask out optional channels when not present (before softmax)
        if not has_user_text:
            gates = gates.clone()
            gates[:, :, 2] = float('-inf')
        if not has_function:
            gates = gates.clone()
            gates[:, :, 3] = float('-inf')
        
        # Softmax over modalities for normalization
        gates = torch.softmax(gates, dim=-1)  # (B, T, 4)
        
        # Stack normalized modalities for weighted sum: (B, T, 4, D)
        stacked = torch.stack([agent_normed, audio_normed, text_normed, function_normed], dim=2)
        
        # Apply gates (broadcast) and sum over modalities
        fused = (stacked * gates.unsqueeze(-1)).sum(dim=2)  # (B, T, D)
        
        return fused


class GatedFusionGMU(FusionModule):
    """
    Gated Multimodal Unit inspired by Arevalo et al. (2017).
    
    PER-DIMENSION gating: each dimension can attend to different modalities.
    Uses softmax over modalities for numerically stable normalization.
    Supports 4 modalities (agent, audio, user_text, function); optional channels use zeros and are masked.
    
    All streams are LayerNorm'd before gate computation and fusion.
    Gate output shape: (B, T, 4, D) with softmax over the modality dimension.
    """
    
    def __init__(self, hidden_dim: int, num_modalities: int = 4):
        super().__init__(hidden_dim)
        self.num_modalities = num_modalities
        
        # LayerNorm for each stream (normalizes distributions before gate computation)
        self.ln_agent = nn.LayerNorm(hidden_dim)
        self.ln_audio = nn.LayerNorm(hidden_dim)
        self.ln_text = nn.LayerNorm(hidden_dim)
        self.ln_function = nn.LayerNorm(hidden_dim)
        
        # Per-modality gate projections (output logits, not activations)
        self.gate_agent = nn.Linear(hidden_dim * num_modalities, hidden_dim)
        self.gate_audio = nn.Linear(hidden_dim * num_modalities, hidden_dim)
        self.gate_text = nn.Linear(hidden_dim * num_modalities, hidden_dim)
        self.gate_function = nn.Linear(hidden_dim * num_modalities, hidden_dim)
        
        # Initialize for uniform weighting (all logits = 0 → equal softmax weights)
        self._init_uniform()
    
    @property
    def uses_learned_gating(self) -> bool:
        return True
    
    def _init_uniform(self):
        """Initialize gate networks for uniform initial weighting."""
        for gate in [self.gate_agent, self.gate_audio, self.gate_text, self.gate_function]:
            nn.init.zeros_(gate.weight)
            nn.init.zeros_(gate.bias)
    
    def forward(
        self,
        agent_text_embeds: torch.Tensor,
        user_audio_embeds: torch.Tensor,
        user_text_embeds: Optional[torch.Tensor] = None,
        function_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, D = agent_text_embeds.shape
        
        has_user_text = user_text_embeds is not None
        has_function = function_embeds is not None
        text_input = user_text_embeds if has_user_text else torch.zeros_like(agent_text_embeds)
        function_input = function_embeds if has_function else torch.zeros_like(agent_text_embeds)
        
        # Apply LayerNorm to each stream before gate computation
        agent_normed = self.ln_agent(agent_text_embeds)
        audio_normed = self.ln_audio(user_audio_embeds)
        text_normed = self.ln_text(text_input)
        function_normed = self.ln_function(function_input)
        
        # Concatenate normalized inputs for gate computation
        concat = torch.cat([agent_normed, audio_normed, text_normed, function_normed], dim=-1)
        
        # Compute per-modality gate logits: each is (B, T, D)
        logits_agent = self.gate_agent(concat)
        logits_audio = self.gate_audio(concat)
        logits_text = self.gate_text(concat)
        logits_function = self.gate_function(concat)
        
        # Stack logits: (B, T, 4, D)
        stacked_logits = torch.stack([logits_agent, logits_audio, logits_text, logits_function], dim=2)
        
        # Mask out optional channels when not present (before softmax)
        if not has_user_text:
            stacked_logits = stacked_logits.clone()
            stacked_logits[:, :, 2, :] = float('-inf')
        if not has_function:
            stacked_logits = stacked_logits.clone()
            stacked_logits[:, :, 3, :] = float('-inf')
        
        # Softmax over modalities (dim=2) for stable normalization
        gates = torch.softmax(stacked_logits, dim=2)  # (B, T, 4, D)
        
        # Stack normalized embeddings: (B, T, 4, D)
        stacked_embeds = torch.stack([agent_normed, audio_normed, text_normed, function_normed], dim=2)
        
        # Fuse with element-wise gating
        fused = (gates * stacked_embeds).sum(dim=2)  # (B, T, D)
        
        return fused


def create_fusion_module(
    fuse_method: Optional[Union[str, FuseMethod]],
    hidden_dim: int,
    agent_text_weight: float = 1.0,
    user_audio_weight: float = 1.0,
    user_text_weight: float = 1.0,
    function_weight: float = 1.0,
) -> FusionModule:
    """
    Factory function to create the appropriate fusion module.
    
    Args:
        fuse_method: One of None, "add", "concat", "gated_simple", "gated_gmu"
        hidden_dim: Hidden dimension of the embeddings
        agent_text_weight: Weight for agent text (used by add/concat methods only)
        user_audio_weight: Weight for user audio (used by add/concat methods only)
        user_text_weight: Weight for user text (used by add/concat methods only)
        function_weight: Weight for function channel (used by add/concat only; optional channel)
    
    Returns:
        Instantiated fusion module
    
    Notes:
        - All fusion modules accept optional function_embeds; when None, that channel
          contributes nothing (zeros + mask for gated methods).
        - For gated methods (gated_simple, gated_gmu), the weight parameters are
          ignored as the gating mechanism learns the optimal combination.
        - When using gated methods, you should also bypass channel embeddings,
          tie_and_roll, and any manual embedding weight parameters in the model.
    """
    if fuse_method is None or fuse_method == "add":
        logging.info(f"Using AddFusion with weights: agent={agent_text_weight}, "
                     f"audio={user_audio_weight}, text={user_text_weight}, function={function_weight}")
        return AddFusion(
            hidden_dim=hidden_dim,
            agent_text_weight=agent_text_weight,
            user_audio_weight=user_audio_weight,
            user_text_weight=user_text_weight,
            function_weight=function_weight,
        )
    elif fuse_method == "concat":
        logging.info(f"Using ConcatFusion with weights: agent={agent_text_weight}, "
                     f"audio={user_audio_weight}, text={user_text_weight}, function={function_weight}")
        return ConcatFusion(
            hidden_dim=hidden_dim,
            agent_text_weight=agent_text_weight,
            user_audio_weight=user_audio_weight,
            user_text_weight=user_text_weight,
            function_weight=function_weight,
        )
    elif fuse_method == "gated_simple":
        logging.info(f"Using GatedFusionSimple with hidden_dim={hidden_dim} "
                     "(learned gating, optional function channel)")
        return GatedFusionSimple(hidden_dim=hidden_dim)
    elif fuse_method == "gated_gmu":
        logging.info(f"Using GatedFusionGMU with hidden_dim={hidden_dim} "
                     "(learned gating, optional function channel)")
        return GatedFusionGMU(hidden_dim=hidden_dim)
    else:
        raise ValueError(
            f"Unknown fuse_method: '{fuse_method}'. "
            f"Expected one of: None, 'add', 'concat', 'gated_simple', 'gated_gmu'"
        )

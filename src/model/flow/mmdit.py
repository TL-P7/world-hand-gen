"""MMDiT block: hybrid attention, joint self-attention, and FFN.

Each block applies, in order::

    hybrid-attn(latent + condition) -> own self-attn
    joint self-attention (MANO + Camera)
    dual-stream hybrid attention (own latent + stream conditions)
    FFN

The MANO and Camera streams keep independent AdaLN-Zero modulation driven by
the shared flow-time embedding. Hybrid attention uses latent-only queries and
combines latent and condition tokens as keys and values. The final dual-stream
attention does not mix MANO and Camera tokens: each stream queries its own
latent tokens and attends to its own latent K/V plus the corresponding
condition tokens and bbox.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .attention import Modulation, SelfAttention, modulate
from .state import CameraFlowState, ConditionTokens, ManoFlowState
from .time import ScalarTimeEncoder


class FFN(nn.Module):
    def __init__(self, model_dim: int, ff_multiplier: int = 4) -> None:
        super().__init__()
        hidden = model_dim * ff_multiplier
        self.network = nn.Sequential(
            nn.Linear(model_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, model_dim),
        )

    def forward(self, value: Tensor) -> Tensor:
        return self.network(value)


class MMDiTBlock(nn.Module):
    def __init__(self, model_dim: int, heads: int, ff_multiplier: int = 4) -> None:
        super().__init__()
        # Segment 0 is a hybrid attention stage: each latent stream queries
        # itself while also attending to the full condition sequence.
        self.cross_mano = SelfAttention(model_dim, heads)
        self.cross_camera = SelfAttention(model_dim, heads)
        # Segment 0 own self-attention and Segment 1 joint self-attention.
        self.self_mano = SelfAttention(model_dim, heads)
        self.self_camera = SelfAttention(model_dim, heads)
        self.self_joint = SelfAttention(model_dim, heads)
        # Segment 2 keeps the two streams separate while adding stream-specific
        # latent tokens and conditions to each attention K/V sequence.
        self.dual_mano = SelfAttention(model_dim, heads)
        self.dual_camera = SelfAttention(model_dim, heads)
        self.ff_mano = FFN(model_dim, ff_multiplier)
        self.ff_camera = FFN(model_dim, ff_multiplier)
        # Five AdaLN-Zero slots per stream: segment-0 cross, segment-0 self,
        # segment-1 joint self, segment-2 dual attention, and FFN.
        self.mod_mano = Modulation(model_dim, 5)
        self.mod_camera = Modulation(model_dim, 5)
        self.mano_norms = nn.ModuleList([nn.LayerNorm(model_dim) for _ in range(5)])
        self.camera_norms = nn.ModuleList([nn.LayerNorm(model_dim) for _ in range(5)])

    def forward(
        self,
        mano: Tensor,  # [B,2T,D]
        camera: Tensor,  # [B,T,D]
        mano_positions: Tensor,  # [B,2T]
        camera_positions: Tensor,  # [B,T]
        condition_kv: Tensor,  # [B,C_total,D]
        condition_positions: Tensor,  # [B,C_total]
        hamer: Tensor,  # [B,2T,D]
        vggt: Tensor,  # [B,17T,D]
        bbox: Tensor,  # [B,2T,D]
        hamer_positions: Tensor,  # [B,2T]
        vggt_positions: Tensor,  # [B,17T]
        bbox_positions: Tensor,  # [B,2T]
        flow_embedding: Tensor,  # [B,D]
        keypoints: Tensor | None = None,  # [B,2T,D]
        keypoint_positions: Tensor | None = None,  # [B,2T]
        mano_condition_kv: Tensor | None = None,
        mano_condition_positions: Tensor | None = None,
        camera_condition_kv: Tensor | None = None,
        camera_condition_positions: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        m_shift, m_scale, m_gate = self.mod_mano(flow_embedding)  # [B,5,D]
        c_shift, c_scale, c_gate = self.mod_camera(flow_embedding)

        if keypoints is None:
            keypoints = bbox.new_zeros(bbox.shape)
        if keypoint_positions is None:
            keypoint_positions = bbox_positions
        if mano_condition_kv is None:
            mano_condition_kv = torch.cat([hamer, keypoints, bbox], dim=1)
        if mano_condition_positions is None:
            mano_condition_positions = torch.cat(
                [hamer_positions, keypoint_positions, bbox_positions], dim=1
            )
        if camera_condition_kv is None:
            camera_condition_kv = torch.cat([vggt, keypoints, bbox], dim=1)
        if camera_condition_positions is None:
            camera_condition_positions = torch.cat(
                [vggt_positions, keypoint_positions, bbox_positions], dim=1
            )

        def hybrid(x, attn, norm, shift, scale, gate, pos, condition, condition_pos):
            h = modulate(norm(x), shift, scale)
            key_value = torch.cat([h, condition], dim=1)
            key_positions = torch.cat([pos, condition_pos], dim=1)
            return x + gate.unsqueeze(1) * attn(
                h, key_value, key_value, pos, key_positions
            )

        def own(x, attn, norm, shift, scale, gate, pos):
            h = modulate(norm(x), shift, scale)
            return x + gate.unsqueeze(1) * attn(h, h, h, pos, pos)

        # Segment 0: hybrid then own self-attention.
        mano = hybrid(mano, self.cross_mano, self.mano_norms[0], m_shift[:, 0], m_scale[:, 0], m_gate[:, 0], mano_positions, mano_condition_kv, mano_condition_positions)
        camera = hybrid(camera, self.cross_camera, self.camera_norms[0], c_shift[:, 0], c_scale[:, 0], c_gate[:, 0], camera_positions, camera_condition_kv, camera_condition_positions)
        mano = own(mano, self.self_mano, self.mano_norms[1], m_shift[:, 1], m_scale[:, 1], m_gate[:, 1], mano_positions)
        camera = own(camera, self.self_camera, self.camera_norms[1], c_shift[:, 1], c_scale[:, 1], c_gate[:, 1], camera_positions)

        # Segment 1: joint self-attention without another conditional cross-attn.
        joint_q = torch.cat([
            modulate(self.mano_norms[2](mano), m_shift[:, 2], m_scale[:, 2]),
            modulate(self.camera_norms[2](camera), c_shift[:, 2], c_scale[:, 2]),
        ], dim=1)
        joint_pos = torch.cat([mano_positions, camera_positions], dim=1)
        joint_out = self.self_joint(joint_q, joint_q, joint_q, joint_pos, joint_pos)
        m_out, c_out = joint_out.split([mano.shape[1], camera.shape[1]], dim=1)
        mano = mano + m_gate[:, 2].unsqueeze(1) * m_out
        camera = camera + c_gate[:, 2].unsqueeze(1) * c_out

        # Segment 2: independent hybrid attention for each stream. Each query
        # sees its own latent K/V plus stream-specific conditions and bbox.
        mano_q = modulate(self.mano_norms[3](mano), m_shift[:, 3], m_scale[:, 3])
        mano_kv = torch.cat([mano_q, mano_condition_kv], dim=1)
        mano_kv_positions = torch.cat(
            [mano_positions, mano_condition_positions], dim=1
        )
        mano_out = self.dual_mano(
            mano_q, mano_kv, mano_kv, mano_positions, mano_kv_positions
        )
        mano = mano + m_gate[:, 3].unsqueeze(1) * mano_out

        camera_q = modulate(self.camera_norms[3](camera), c_shift[:, 3], c_scale[:, 3])
        camera_kv = torch.cat([camera_q, camera_condition_kv], dim=1)
        camera_kv_positions = torch.cat(
            [camera_positions, camera_condition_positions], dim=1
        )
        camera_out = self.dual_camera(
            camera_q, camera_kv, camera_kv, camera_positions, camera_kv_positions
        )
        camera = camera + c_gate[:, 3].unsqueeze(1) * camera_out

        # FFN.
        mano = mano + m_gate[:, 4].unsqueeze(1) * self.ff_mano(
            modulate(self.mano_norms[4](mano), m_shift[:, 4], m_scale[:, 4])
        )
        camera = camera + c_gate[:, 4].unsqueeze(1) * self.ff_camera(
            modulate(self.camera_norms[4](camera), c_shift[:, 4], c_scale[:, 4])
        )

        return mano, camera


class MMDiT(nn.Module):
    """Flow-matching MM-DiT predicting MANO and relative camera transforms from noise.

    Two latent streams (MANO and Camera) are denoised jointly; HaMeR, VGGT and
    bbox conditions are injected through Segment 0 cross-attention and the
    final stream-specific hybrid attention. The time step drives per-stream
    AdaLN-Zero modulation.
    """

    def __init__(
        self,
        *,
        local_joints: int = 15,
        rotation_dim: int = 6,
        model_dim: int = 512,
        depth: int = 8,
        heads: int = 8,
        camera_dim: int = 9,
        ff_multiplier: int = 4,
    ) -> None:
        super().__init__()
        self.local_joints = local_joints
        self.rotation_dim = rotation_dim
        self.model_dim = model_dim
        self.mano_dim = 3 + rotation_dim + local_joints * rotation_dim

        self.mano_projection = nn.Linear(self.mano_dim, model_dim)
        self.camera_projection = nn.Linear(camera_dim, model_dim)

        self.mano_type = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.camera_type = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.handedness = nn.Parameter(torch.zeros(1, 2, 1, model_dim))
        self.hamer_type = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.vggt_type = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.bbox_type = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.keypoint_type = nn.Parameter(torch.zeros(1, 1, model_dim))

        self.flow_time = ScalarTimeEncoder(model_dim)
        self.blocks = nn.ModuleList(
            [MMDiTBlock(model_dim, heads, ff_multiplier) for _ in range(depth)]
        )

        self.mano_norm = nn.LayerNorm(model_dim)
        self.camera_norm = nn.LayerNorm(model_dim)
        self.mano_head = nn.Linear(model_dim, self.mano_dim)
        self.camera_head = nn.Linear(model_dim, camera_dim)
        self._init_output_heads()

    def _init_output_heads(self) -> None:
        nn.init.zeros_(self.mano_head.weight)
        nn.init.zeros_(self.mano_head.bias)
        nn.init.zeros_(self.camera_head.weight)
        nn.init.zeros_(self.camera_head.bias)
        # Root rotation 6D identity: first column [1,0,0], second [0,1,0].
        mano_bias = self.mano_head.bias.data
        mano_bias[3] = 1.0
        mano_bias[3 + self.rotation_dim - 2] = 1.0
        for joint in range(self.local_joints):
            offset = 3 + self.rotation_dim + joint * self.rotation_dim
            mano_bias[offset] = 1.0
            mano_bias[offset + self.rotation_dim - 2] = 1.0
        # Camera rotation 6D identity.
        camera_bias = self.camera_head.bias.data
        camera_bias[0] = 1.0
        camera_bias[self.rotation_dim - 2] = 1.0

    def forward(
        self,
        mano: ManoFlowState,
        camera: CameraFlowState,
        flow_time: Tensor,
        conditions: ConditionTokens,
    ) -> tuple[ManoFlowState, CameraFlowState]:
        batch = mano.translation.shape[0]
        time = mano.translation.shape[2]

        motion = torch.cat(
            (mano.translation, mano.root_rotation, mano.local_rotation.flatten(-2)), dim=-1
        )  # [B,2,T,99]
        camera_input = torch.cat((camera.rotation, camera.translation), dim=-1)  # [B,T,9]

        motion_tokens = self.mano_projection(motion) + self.mano_type + self.handedness
        motion_tokens = motion_tokens.reshape(batch, 2 * time, self.model_dim)
        camera_tokens = self.camera_projection(camera_input) + self.camera_type

        frame_index = torch.arange(time, device=motion.device, dtype=motion.dtype)
        mano_positions = frame_index.repeat(2).expand(batch, 2 * time)
        camera_positions = frame_index.expand(batch, time)
        vggt_positions = frame_index.repeat_interleave(17).expand(batch, time * 17)

        hamer_tokens = conditions.hamer + self.hamer_type
        vggt_tokens = conditions.vggt + self.vggt_type
        bbox_tokens = conditions.bbox + self.bbox_type
        keypoint_tokens = conditions.keypoints + self.keypoint_type
        mano_condition_kv = torch.cat((hamer_tokens, keypoint_tokens, bbox_tokens), dim=1)
        mano_condition_positions = torch.cat(
            (mano_positions, mano_positions, mano_positions), dim=1
        )
        camera_condition_kv = torch.cat((vggt_tokens, keypoint_tokens, bbox_tokens), dim=1)
        camera_condition_positions = torch.cat(
            (vggt_positions, mano_positions, mano_positions), dim=1
        )

        flow_embedding = self.flow_time(flow_time)
        for block in self.blocks:
            motion_tokens, camera_tokens = block(
                motion_tokens,
                camera_tokens,
                mano_positions,
                camera_positions,
                mano_condition_kv,
                mano_condition_positions,
                hamer_tokens,
                vggt_tokens,
                bbox_tokens,
                mano_positions,
                vggt_positions,
                mano_positions,
                flow_embedding,
                keypoint_tokens,
                mano_positions,
                mano_condition_kv,
                mano_condition_positions,
                camera_condition_kv,
                camera_condition_positions,
            )

        mano_out = self.mano_head(self.mano_norm(motion_tokens)).reshape(
            batch, 2, time, self.mano_dim
        )
        camera_out = self.camera_head(self.camera_norm(camera_tokens))

        return (
            ManoFlowState(
                translation=mano_out[..., :3],
                root_rotation=mano_out[..., 3:3 + self.rotation_dim],
                local_rotation=mano_out[..., 3 + self.rotation_dim:].reshape(
                    batch, 2, time, self.local_joints, self.rotation_dim
                ),
            ),
            CameraFlowState(
                rotation=camera_out[..., :self.rotation_dim],
                translation=camera_out[..., self.rotation_dim:],
            ),
        )

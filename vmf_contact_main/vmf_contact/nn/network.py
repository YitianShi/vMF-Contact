import cv2
import einops
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .attention import FFWRelativeSelfCrossAttentionModule
from .position_encodings import (
    LearnedAbsolutePositionEncoding3Dv2,
    RotaryPositionEncoding3D,
)
from .pointr import Fold
from .util import match
from openpoints.models import build_model_from_cfg


class vmfContact(nn.Module):

    def __init__(
        self,
        args,
        # input
        image_size=(480, 640),
        scale=7 / 8,
        pcd_with_rgb=False,
        # encoder parameters
        q_num=200,
        embedding_dim=384,
        num_attn_heads=4,
        query_self_attn=True,  # Whether to use self-attention before cross-attention
        num_query_cross_attn_layers=4,
        pos_embedd_type="rel",
        # output
        fine_sampling_ball_diameter=0.16,
        # uncertainty parameters
        prob_baseline=None,
        # diffusion
        diffusion=False,
    ):
        super().__init__()

        self.image_size = (int(scale * image_size[0]), int(scale * image_size[1]))
        self.pcd_dim = 6 if pcd_with_rgb else 3
        self.sampling_ball_diameter_pyramid = [
            fine_sampling_ball_diameter,
            fine_sampling_ball_diameter / 2.0,
            fine_sampling_ball_diameter / 4.0,
            fine_sampling_ball_diameter / 8.0,
        ]
        self.embedding_dim = embedding_dim

        # 3D relative positional embeddings
        assert pos_embedd_type in ["rel", "abs"]
        if pos_embedd_type == "rel":
            self.relative_pe_layer = RotaryPositionEncoding3D(embedding_dim)
        elif pos_embedd_type == "abs":
            self.relative_pe_layer = LearnedAbsolutePositionEncoding3Dv2(embedding_dim)

        # Contact point learnable initial features
        self.contact_points_embed = nn.Embedding(1, embedding_dim)

        # Point backbone
        self.point_backbone = build_model_from_cfg(args.point_backbone_cfgs.model)
        encoder_args = args.point_backbone_cfgs.model["encoder_args"]
        
        if "strides" in encoder_args.keys():
            strides = [x for x in encoder_args["strides"] if x != 1]
            first_channel = args.point_backbone_cfgs.model["encoder_args"]["width"]
            n_blocks = len(strides)
        elif "n_blocks" in encoder_args.keys():
            first_channel = encoder_args["channels"]
            n_blocks = encoder_args["n_blocks"] - 1

        channels = [first_channel * 2 ** (i+1) for i in range(n_blocks)]
        self.map_features = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(channel, channel, 1),
                nn.BatchNorm1d(channel),
                nn.Conv1d(channel, embedding_dim, 1)            
                ) 
            for channel in channels
            ])

        # Query learnable features
        self.query_embed = nn.Embedding(q_num, embedding_dim)

        # Query cross-attention to contact points
        if num_query_cross_attn_layers > 0:
            self.query_cross_attn_pcd = FFWRelativeSelfCrossAttentionModule(
                embedding_dim,
                num_attn_heads,
                num_query_cross_attn_layers,
                self_attn=query_self_attn,
                use_adaln=False,
            )
        else:
            self.query_cross_attn_pcd = None

        # Contact point position prediction
        self.contact_point_fnn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 3),
        )

        # Grasp prediction heads
        self.baseline_dim = 4 if prob_baseline is not None else 3
        self.bin_num = 19

        # Gripper rotation (quaternion) and binary opening width
        self.rotation_dim = 4
        
        self.diffusion = diffusion
        if not diffusion:
            self.orientation_head = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim // 2),
                nn.ReLU(),
                nn.Linear(embedding_dim // 2, embedding_dim // 2),
                nn.ReLU(),
                nn.Linear(embedding_dim // 2, embedding_dim),
                nn.ReLU(),
                nn.Linear(embedding_dim, self.baseline_dim + self.bin_num + 2),
            )

        self.increase_dim = nn.Sequential(
            nn.Conv1d(embedding_dim, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, 1024, 1)
        )
        self.reduce_map = nn.Linear(embedding_dim + 1027, embedding_dim)
        self.foldingnet = Fold(embedding_dim, step=4)

    def forward(self, current_pcd, gt_cp_batch=None, debug=False, reconstruct=True):
        if current_pcd.dim() == 2:
            current_pcd = current_pcd.unsqueeze(0)

        self.debug = debug and self.training
        self.batch_size = current_pcd.shape[0]
        self.device = current_pcd.device

        # Get local features of the point groups
        contact_pcd_pos, contact_pcd_features, sample_ids = self.point_backbone(current_pcd)
        for i in range(len(contact_pcd_features)):
            contact_pcd_features[i] = self.map_features[i](contact_pcd_features[i])

        num_cp = contact_pcd_pos[0].shape[1]
        contact_pcd_i = contact_pcd_pos[0] # B M 3
        contact_pcd_feature = contact_pcd_features[0] # B C M

        if self.query_cross_attn_pcd is not None:
             # Initialize query features
            kv_feature = contact_pcd_features[-1].permute(2, 0, 1) # Q B C
            # Embed contact points
            embed = self.contact_points_embed # B 1 C
            contact_pcd_features_embed = embed.weight.unsqueeze(0).repeat(
                contact_pcd_i.shape[1], self.batch_size, 1
            ) # M B C

            contact_pcd_feature = (
                contact_pcd_feature.permute(2, 0, 1) + contact_pcd_features_embed
            ) # M B C
            # The query cross-attends to contact point features (point features)
            contact_pcd_feature = self._compute_cross_attn_features(
                contact_pcd_feature,
                kv_feature,
                self.relative_pe_layer(contact_pcd_i),
                None,
            )[-1]
            
            contact_pcd_feature = contact_pcd_feature.permute(1,0,2).contiguous() # B M C
        else:
            contact_pcd_feature = contact_pcd_feature.permute(0,2,1).contiguous() 

        if reconstruct:
            global_feature = self.increase_dim(contact_pcd_feature.transpose(1, 2)).transpose(1,2) # B M 1024
            global_feature = torch.max(global_feature, dim=1)[0] # B 1024

            rebuild_feature = torch.cat([
                global_feature.unsqueeze(1).expand(-1, num_cp, -1),
                contact_pcd_feature,
                contact_pcd_i], dim=-1)  # B M 1027 + C

            rebuild_feature = self.reduce_map(rebuild_feature.reshape(self.batch_size*num_cp, -1)) # BM C

            relative_xyz = self.foldingnet(rebuild_feature).reshape(self.batch_size, num_cp, 3, -1)
            #contact_pcd_feature = contact_pcd_feature.reshape(self.batch_size, num_cp, -1) # B M C
            reconstructed_pcds = (contact_pcd_i.unsqueeze(-1) + relative_xyz)
            reconstructed_pcds = einops.rearrange(reconstructed_pcds, "b m c n -> b (m n) c")
        else:
            reconstructed_pcds = None

        # decode query position from query features
        cp = self.contact_point_fnn(contact_pcd_feature) + contact_pcd_i

        # Predict the grasps (contact point, baseline vector, bin scores)
        baseline, bin_score, grasp_width, graspness = self._predict_grasp(contact_pcd_feature) if not self.diffusion else None, None, None, None

        # match the ground-truth contact points to the query positions
        if gt_cp_batch is not None:
            # During training, match the ground-truth contact points to the sampled positions
            anchor, matched_ind = self.match(contact_pcd_i, gt_cp_batch)

            sample_ids = [sample_ids[0][i][matched_ind[i][1]] for i in range(self.batch_size)]
        else:
            # During inference, use the last-level query points as anchors to sample potential contact points
            anchor = cp

        return {
            # Grasps
            "contact_point": cp,
            "baseline": baseline,
            "bin_score": bin_score,
            "grasp_width": grasp_width,
            "graspness": graspness,
            # matches
            #"group_point_pos": group_point_pos,
            "anchor": anchor,
            "matched_ind": matched_ind if gt_cp_batch is not None else None,
            # Auxiliary outputs used to compute the loss or for visualization
            "position": cp,
            "cp_features": contact_pcd_feature,
            "sample_ids": sample_ids,
            # Reconstruction
            "reconstructed_pcds": reconstructed_pcds
        }

    def match(self, cp_position, gt_cp_batch, hungarian=False):
        # Match the ground-truth contact points to the last-level query positions
        matched_indices = match(gt_cp_batch, cp_position, hungarian=hungarian)
        anchor = []
        for bt in range(len(gt_cp_batch)):
            # Fetch matched ground-truth contact points
            matched_anch = gt_cp_batch[bt][matched_indices[bt][0]]
            anchor.append(matched_anch)
        return anchor, matched_indices

    def _compute_visual_features(
        self, visible_rgb, visible_pcd, num_cameras, visible_rgb_features=None
    ):
        """Compute visual features at different scales and their positional embeddings.
        Args:
            visible_rgb: (B, ncam, 3, H, W), pixel intensities
            visible_pcd: (B, ncam, 3, H, W), positions
            num_cameras: Number of cameras
        Returns:
            rgb_feats_pyramid: [(B, ncam, F, H_i, W_i)]
            pcd_pyramid: [(B, ncam * H_i * W_i, 3)]
        """

        # Pass each view independently through backbone
        if visible_rgb_features is None:
            visible_rgb = einops.rearrange(
                visible_rgb, "bt ncam c h w -> (bt ncam) c h w"
            )
            visible_rgb_features = self.get_vlm_features(visible_rgb)

        visible_pcd = einops.rearrange(visible_pcd, "bt ncam c h w -> (bt ncam) c h w")

        visible_rgb_features_pyramid = []
        visible_rgb_pos_pyramid = []
        visible_pcd_pyramid = []

        # Pass visual features through feature pyramid network
        visible_rgb_features = self.feature_pyramid(visible_rgb_features)

        for i in range(self.num_sampling_level):
            visible_rgb_features_i = visible_rgb_features[self.feature_map_pyramid[i]]
            visible_pcd_i = F.interpolate(
                visible_pcd,
                scale_factor=1.0 / self.downscaling_factor_pyramid[i],
                mode="bilinear",
            )
            h, w = visible_pcd_i.shape[-2:]

            # visualize visible_pcd_i as rgb image
            if self.debug:
                visible_pcd_depth = visible_pcd_i[0, 2].clone().cpu().numpy()
                visible_pcd_depth = (
                    (visible_pcd_depth - visible_pcd_depth.min())
                    / (visible_pcd_depth.max() - visible_pcd_depth.min())
                    * 255
                )
                visible_pcd_depth = visible_pcd_depth.astype(np.uint8)
                visible_pcd_depth = cv2.cvtColor(visible_pcd_depth, cv2.COLOR_GRAY2RGB)
                cv2.imwrite("pic_debug/visible_pcd_depth.jpg", visible_pcd_depth)

            visible_pcd_i = einops.rearrange(
                visible_pcd_i, "(bt ncam) c h w -> bt (ncam h w) c", ncam=num_cameras
            )
            visible_rgb_pos_i = self.relative_pe_layer(visible_pcd_i)
            visible_rgb_features_i = einops.rearrange(
                visible_rgb_features_i,
                "(bt ncam) c h w -> bt ncam c h w",
                ncam=num_cameras,
            )

            visible_rgb_features_pyramid.append(visible_rgb_features_i)
            visible_rgb_pos_pyramid.append(visible_rgb_pos_i)
            visible_pcd_pyramid.append(visible_pcd_i)

        return (
            visible_rgb_features_pyramid,
            visible_rgb_pos_pyramid,
            visible_pcd_pyramid,
        )  # , intermediate_layers

    def _compute_cross_attn_features(
        self,
        query_features,
        kv_features,
        query_pos,
        kv_pos,
        need_attention_weights=False,
    ):
        """The query cross-attends to context features (visual features, instruction features,
        and current gripper position)."""
        attn_layers = self.query_cross_attn_pcd

        query_features, attention_weights = attn_layers(
            query=query_features,
            context=kv_features,
            query_pos=query_pos,
            context_pos=kv_pos,
            need_attention_weights=need_attention_weights,
        )

        if need_attention_weights:
            return query_features, attention_weights
        return query_features


    def encode_images(self, rgb, pcd):
        """
        Compute visual features/pos embeddings at different scales.

        Args:
            - rgb: (B, ncam, 3, H, W), pixel intensities
            - pcd: (B, ncam, 3, H, W), positions

        Returns:
            - rgb_feats_pyramid: [(B, ncam, F, H_i, W_i)]
            - pcd_pyramid: [(B, ncam * H_i * W_i, 3)]
        """
        num_cameras = rgb.shape[1]

        # Pass each view independently through backbone
        rgb = einops.rearrange(rgb, "bt ncam c h w -> (bt ncam) c h w")
        rgb = self.preprocess(rgb)
        rgb_features = self.backbone(rgb)

        # Pass visual features through feature pyramid network
        rgb_features = self.feature_pyramid(rgb_features)

        # Treat different cameras separately
        pcd = einops.rearrange(pcd, "bt ncam c h w -> (bt ncam) c h w")

        rgb_feats_pyramid = []
        pcd_pyramid = []
        for i in range(self.num_sampling_level):
            # Isolate level's visual features
            rgb_features_i = rgb_features[self.feature_map_pyramid[i]]

            # Interpolate xy-depth to get the locations for this level
            pcd_i = F.interpolate(
                pcd,
                scale_factor=1.0 / self.downscaling_factor_pyramid[i],
                mode="bilinear",
            )

            # Merge different cameras for clouds, separate for rgb features
            h, w = pcd_i.shape[-2:]
            pcd_i = einops.rearrange(
                pcd_i, "(bt ncam) c h w -> bt (ncam h w) c", ncam=num_cameras
            )
            rgb_features_i = einops.rearrange(
                rgb_features_i, "(bt ncam) c h w -> bt ncam c h w", ncam=num_cameras
            )

            rgb_feats_pyramid.append(rgb_features_i)
            pcd_pyramid.append(pcd_i)

        return rgb_feats_pyramid, pcd_pyramid

    def _predict_grasp(self, features):

        # Predict rotation and gripper opening
        orientation = self.orientation_head(features)
        baseline = orientation[..., : self.baseline_dim]

        bin_score = orientation[..., self.baseline_dim : self.baseline_dim + self.bin_num]
        grasp_width = orientation[..., -2]
        graspness = orientation[..., -1]

        return baseline, bin_score, grasp_width, graspness


def sample_contact_points_uniform_sphere(center, radius, bounds, num_points=1000):
    """Sample points uniformly within a sphere through rejection sampling."""
    contact_points = np.empty((0, 3))
    while contact_points.shape[0] < num_points:
        points = sample_contact_points_uniform_cube(bounds, num_points)
        l2 = np.linalg.norm(points - center, axis=1)
        contact_points = np.concatenate([contact_points, points[l2 < radius]])
    contact_points = contact_points[:num_points]
    return contact_points


def sample_contact_points_uniform_cube(bounds, num_points=1000):
    x = np.random.uniform(bounds[0][0], bounds[1][0], num_points)
    y = np.random.uniform(bounds[0][1], bounds[1][1], num_points)
    z = np.random.uniform(bounds[0][2], bounds[1][2], num_points)
    contact_points = np.stack([x, y, z], axis=1)
    return contact_points

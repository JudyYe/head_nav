"""vision tower"""

import einops
import numpy as np
import torch
import torch.nn as nn


class VisionTower(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_dim = config.token_dim
        self.vision_tower = build_vision_tower(config, delay_load=True)
        self.vision_tower.load_model()
        # freeze vision tower
        for param in self.vision_tower.parameters():
            param.requires_grad = False
        self.mm_projector = nn.Linear(self.vision_tower.hidden_size, config.hidden_size)

    def encode_images(self, images):
        tokens = self.vision_tower(images)  # (BT, HW, D)
        tokens = self.mm_projector(tokens)
        num_tokens_pre_frame = tokens.shape[1]
        tokens = einops.rearrange(tokens, "(b t) s d -> b t s d", b=self.b)
        assert tokens.shape == torch.Size(
            [self.b, self.t, num_tokens_pre_frame, self.token_dim]
        ), tokens.shape
        return tokens

    def forward(self, images):
        return self.visual_to_tokens(images)

    def visual_to_tokens(self, images):
        input_type = getattr(self.config, "input_type", "video")
        self.b, self.t, self.c, self.h, self.w = images.shape

        if input_type == "image":
            return self.images_to_tokens(images)
        elif input_type == "video":
            visual_tokens = self.videos_to_tokens(images)
            return visual_tokens

    def images_to_tokens(self, images):
        if type(images) is list or images.ndim == 5:
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [x.flatten(0, 1) for x in image_features]
        else:
            image_features = self.encode_images(images)

        return image_features

    def videos_to_tokens(self, images):
        assert images.ndim == 5, "multiple videos per sample not supported yet"
        b, t, c, h, w = images.shape
        images = einops.rearrange(images, "b t c h w -> (b t) c h w")
        assert images.shape == torch.Size([b * t, c, h, w])
        tokens = self.encode_images(images)  # (b t) s d
        d = tokens.shape[-1]
        assert tokens.shape == torch.Size([b, t, 256, d])

        video_arch = getattr(self.config, "video_arch", "temporal")
        if video_arch == "all":
            tokens = einops.rearrange(tokens, "b t s d -> b (t s) d")
            assert tokens.shape == torch.Size([b, t * 256, d])
            return tokens
        if video_arch == "temporal":
            tokens = einops.reduce(tokens, "b t s d -> b t d", "mean")
        elif video_arch == "spatial":
            tokens = einops.reduce(tokens, "b t s d -> b s d", "mean")
        elif video_arch == "temporal_spatial":
            t_tokens = einops.reduce(tokens, "b t s d -> b t d", "mean")
            s_tokens = einops.reduce(tokens, "b t s d -> b s d", "mean")
            tokens = torch.cat([t_tokens, s_tokens], dim=1)
        elif video_arch == "temporal_spatial_pool" or video_arch == "spatial_pool":
            pool_size = 2
            selected_frames = np.round(
                np.linspace(0, tokens.shape[1] - 1, pool_size * pool_size)
            ).astype(int)
            s_tokens = tokens[:, selected_frames, ...]
            assert s_tokens.shape == torch.Size([b, pool_size * pool_size, 256, d])
            s_tokens = einops.rearrange(
                s_tokens, "b t (h w) d -> (b t) d h w", h=16, w=16
            )
            assert s_tokens.shape == torch.Size([b * pool_size * pool_size, d, 16, 16])
            s_tokens = nn.functional.avg_pool2d(s_tokens, kernel_size=pool_size)
            assert s_tokens.shape == torch.Size([b * pool_size * pool_size, d, 8, 8])
            s_tokens = einops.rearrange(s_tokens, "(b t) d h w -> b (t h w) d", b=b)
            assert s_tokens.shape == torch.Size([b, 4 * 8 * 8, d])

            if video_arch == "temporal_spatial_pool":
                t_tokens = einops.reduce(tokens, "b t s d -> b t d", "mean")
                assert t_tokens.shape == torch.Size([b, t, d])
                tokens = torch.cat([t_tokens, s_tokens], dim=1)
                assert tokens.shape == torch.Size([b, t + 4 * 8 * 8, d])
                # output dim: (b, t + 4 * 8 * 8, d)
            elif video_arch == "spatial_pool":
                tokens = s_tokens
                assert tokens.shape == torch.Size([b, 4 * 8 * 8, d])
        else:
            raise ValueError(f"unknown video arch {video_arch}")

        return tokens


class DINOV2VisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()
        self.is_loaded = False
        self.vision_tower_name = vision_tower

        if not delay_load:
            self.load_model()

    def load_model(self):
        self.vision_tower = torch.hub.load(
            "facebookresearch/dinov2", self.vision_tower_name
        )
        self.vision_tower.eval()
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    @torch.no_grad()
    def forward(self, images):
        return self.vision_tower.forward_features(images)["x_norm_patchtokens"]

    @property
    def hidden_size(self):
        return self.vision_tower.embed_dim


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(
        vision_tower_cfg,
        "mm_vision_tower",
        getattr(vision_tower_cfg, "vision_tower", None),
    )
    if "dino" in vision_tower:
        return DINOV2VisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f"Unknown vision tower: {vision_tower}")

import os.path as osp

import pytorch_lightning as pl
import torch
import torch.nn as nn
import wandb

from model.backbone import VisionTower
from model.head import TransformerCVAE
from nnutils import cam_utils, image_utils
from nnutils.visualizer import Visualizer


class CameraCVAE(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.Tin = config.Tin
        self.Tout = config.Tout
        self.viz = Visualizer()

        self.backbone = VisionTower(config.backbone)
        # freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.camera_encoder = nn.Linear(4 * 4, config.cond_dim)
        self.goal_encoder = nn.Linear(2, config.cond_dim)
        self.traj_head = TransformerCVAE(
            x_dim=self.Tout * 9,
            latent_dim=config.latent_dim,
            cond_dim=config.cond_dim,
            hidden_dim=config.hidden_dim,
            config=config,
        )

    def vis_step(self, batch, output, pref=""):
        save_pref = osp.join(
            self.config.exp_dir, "wandb", f"{self.global_step:06d}_{pref}"
        )

        camera_next_pred = output["camera_next"].reshape(-1, self.Tout, 4, 4)
        camera_next_gt = batch["camera_next"].reshape(-1, self.Tout, 4, 4)

        image_list, _ = self.viz.vis_c1Tc2_list([camera_next_gt, camera_next_pred])
        image_utils.save_gif(image_list, save_pref + "_camera")
        image_utils.save_images(image_list[0], save_pref + "_camera")
        self.logger.log_metrics(
            {
                "vis/camera_recon_vid": wandb.Video(save_pref + "_camera.gif"),
                "vis/camera_recon_img": wandb.Image(save_pref + "_camera.png"),
            },
            self.global_step,
        )

        video = self.viz.frame_videos(batch["images"], batch["images_next"])
        image_utils.save_gif(video.transpose(0, 1), save_pref + "_video", fps=5)
        self.logger.log_metrics(
            {"vis/video": wandb.Video(save_pref + "_video.gif")}, self.global_step
        )
        video = self.viz.vis_goals(video, batch["goal2D_all"])
        image_utils.save_gif(video.transpose(0, 1), save_pref + "_video_goal", fps=5)
        self.logger.log_metrics(
            {"vis/video_goal": wandb.Video(save_pref + "_video_goal.gif")},
            self.global_step,
        )

        # sample
        sample_list = []
        for s in range(3):
            out, _ = self.sample(batch)  # (c1Tc2)  # (B, T, 4*4)
            sample_list.append(out)

        image_list, _ = self.viz.vis_c1Tc2_list(
            sample_list, color_ind=1
        )  # (N, T, 4, 4) -> (image list and meshes)
        image_utils.save_gif(image_list, f"{save_pref}_sample")
        image_utils.save_images(image_list[0], f"{save_pref}_sample")
        self.logger.log_metrics(
            {
                "vis/sample_vid": wandb.Video(f"{save_pref}_sample.gif"),
                "vis/sample_img": wandb.Image(f"{save_pref}_sample.png"),
            },
            self.global_step,
        )

    def encode_z(self, batch):
        camera_next = batch["camera_next"]
        cond = self.get_cond(batch)
        mean, log_var = self.traj_head.encoder(cond, camera_next)

        return mean, {"log_var": log_var}

    def forward(self, batch, z=None):
        cond = self.get_cond(batch)
        camera_next, results = self.traj_head(
            cond, z=z, is_human=batch.get("is_human", None)
        )

        return camera_next, results

    def sample(self, batch, z=None):
        return self(batch, z=z)

    def get_cond(self, batch):
        images = batch["images"]  # (B, Tin, C, H, W)
        camera_prev = batch["camera_prev"]  # (B, Tin-1, 4x4)
        assert (
            camera_prev.shape[1] == self.Tin - 1
        ), f"camera_prev shape error: {camera_prev.shape[1]} != {self.Tin - 1}"
        # normalize translation
        if self.Tin > 1:
            camera_prev = camera_prev.reshape(-1, 4, 4)
            tsl = camera_prev[:, :3, 3]
            tsl = tsl / (torch.norm(tsl, dim=-1, keepdim=True) + 1e-3)

            camera_prev[:, :3, 3] = tsl
            camera_prev = camera_prev.reshape(-1, self.Tin - 1, 4 * 4)
        goal = batch["goal_2D"]  # (B, 2)

        visual_feat = self.backbone(images)  # (B, N, D=512)
        camera_feat = self.camera_encoder(camera_prev)  # (B, T_in, D=512)
        # print('camera_feat', camera_feat.shape, camera_prev.shape, visual_feat.shape)
        goal_feat = self.goal_encoder(goal).unsqueeze(1)

        # cond = visual_feat + camera_feat + goal_feat
        cond = torch.cat([visual_feat, camera_feat, goal_feat], dim=1)
        return cond

    def forward_step(self, batch, resample=True):
        camera_next_gt = batch["camera_next"]
        cond = self.get_cond(batch)

        camera_next, results = self.traj_head.step(
            cond,
            gt=camera_next_gt,
            resample=resample,
            is_human=batch.get("is_human", None),
        )

        rtn = {"camera_next": camera_next}
        rtn.update(results)

        loss = self.compute_loss(results)
        return loss, rtn

    def compute_loss(self, results):
        losses = results["losses"]
        kl_loss = losses["kl"]
        recon_loss = losses["recon"]

        cfg = self.config.loss
        loss = kl_loss * cfg.kl + recon_loss * cfg.recon

        if torch.isnan(recon_loss).any() or (recon_loss > 10).any():
            # import pdb; pdb.set_trace()
            # raise ValueError("NaN Loss")
            loss = kl_loss * cfg.kl

        return loss

    def training_step(self, batch, batch_idx):
        loss, results = self.forward_step(batch)

        losses = results["losses"]
        self.logger.log_metrics({"train_loss/Total": loss}, self.global_step)
        self.logger.log_metrics(
            {f"train_loss/{k}": v for k, v in losses.items()}, self.global_step
        )

        return loss

    def validation_step(self, batch, batch_idx):
        loss, results = self.forward_step(batch, resample=False)
        losses = results["losses"]

        self.logger.log_metrics({"val_loss/Total": loss}, self.global_step)
        self.logger.log_metrics(
            {f"val_loss/{k}": v for k, v in losses.items()}, self.global_step
        )

        self.vis_step(batch, results)

        return loss

    def configure_optimizers(self):
        mode = self.config.get("mode", "ft")
        if mode == "ft":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.lr)
        elif mode == "adapt":
            # don't change layer deccam or deccam_human
            params = []
            for name, param in self.named_parameters():
                if "deccam" in name or "deccam_human" in name:
                    param.requires_grad = False
                    print(f"freeze {name}")
                else:
                    params.append(param)
            optimizer = torch.optim.AdamW(params, lr=self.config.lr)

        return optimizer

    def rollout(self, c1Tc2):
        B, T_1 = c1Tc2.shape[:2]
        c1Tc2 = c1Tc2.reshape(B, -1, 4, 4)
        cTw = cam_utils.rollout_c1Tc2(c1Tc2)
        return cTw

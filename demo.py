import json
import os
import os.path as osp
from collections import defaultdict

import imageio
import pytorch_lightning as pl
import torch
from hydra import main
from omegaconf import OmegaConf
from tqdm import tqdm

from dataset.robot import RobotData
from model.network import CameraCVAE
from nnutils import geom_utils, image_utils, model_utils
from nnutils.visualizer import Visualizer

device = "cuda:0"


@torch.no_grad()
def eval_model(model, dataloader, cfg, save_dir):
    model.eval()
    model = model.to(device)

    viz = Visualizer()

    result_list = defaultdict(list)
    video_list = []
    for b, batch in enumerate(dataloader):
        if cfg.num > 0 and b >= cfg.num:
            break
        save_pref = os.path.join(save_dir, f"batch{b:04d}_")
        batch = model_utils.to_cuda(batch, device)

        gt = batch["camera_next"]

        cTw_gt = model.rollout(gt)
        cam_gt = geom_utils.inverse_rt_v2(cTw_gt)[..., :3, 3]  # (B, T, 3)

        B = cam_gt.shape[0]
        gt_length = torch.norm(gt.reshape(B, -1, 4, 4)[..., :3, 3], dim=-1).sum(
            -1
        )  # (B, T, )
        result_list["gt_length"].append(gt_length.cpu())

        # sample
        sample_list = []
        for s in range(cfg.S):
            out, _ = model.sample(batch)  # (c1Tc2)  # (B, T, 4*4)
            sample_list.append(out)

            cTw_pred = model.rollout(out)

            # cam center
            cam_pred = geom_utils.inverse_rt_v2(cTw_pred)[..., :3, 3]  # (B, T, 3)
            diff = torch.norm(cam_pred[:, -1] - cam_gt[:, -1], dim=-1)  # (B, 3)
            result_list["diff"].append(diff.cpu())

            error_diff = torch.norm(cam_pred - cam_gt, dim=-1)  # (B, T, 3)
            error_diff = error_diff.mean(-1)
            result_list["error"].append(error_diff.cpu())

            # only compare local differences
            local_diff = out[..., :3, 3] - gt[..., :3, 3]  # (B, T-1, 3)
            local_diff = torch.norm(local_diff, dim=-1)  # (B, T-1)
            local_diff = local_diff.mean(-1)
            result_list["local_diff"].append(local_diff.cpu())

        if cfg.vis:
            image_list, _ = viz.vis_c1Tc2_list(
                sample_list, color_ind=1
            )  # (N, T, 4, 4) -> (image list and meshes)
            image_utils.save_gif(image_list, f"{save_pref}_sample")
            image_utils.save_images(image_list[0], f"{save_pref}_sample")

            videos = torch.cat(
                [batch["images"], batch["images_next"]], dim=1
            )  # (T, 3, H, W)
            videos = viz.vis_goals(videos, batch["goal2D_all"])
            image_utils.save_gif(
                videos.transpose(0, 1), f"{save_pref}_video", fps=2, scale=True
            )
            image_utils.save_images(videos[:, 0], f"{save_pref}_video", scale=True)

        images, image_list = viz.overlay_one_sample(
            batch["images"][:, -1], batch["goal_2D"], [gt] + sample_list, color_ind=0
        )  # (T, 3, H, W)
        video_list.append(images)

    os.makedirs(osp.dirname(save_dir), exist_ok=True)
    imageio.mimsave(
        os.path.join(save_dir + "_pred.mp4"),
        video_list,
        fps=2,
    )
    print("save video to", os.path.join(save_dir, "pred.gif"))

    result_list["diff"] = torch.cat(result_list["diff"], dim=0)
    print("avg diff", result_list["diff"].mean(0))
    # best of diff
    print(
        f"best of {cfg.S} samples",
        result_list["diff"].reshape(-1, cfg.S).min(1).values.mean(0),
    )
    result_list["gt_length"] = torch.cat(result_list["gt_length"], dim=0)
    print("avg length", result_list["gt_length"].mean(0))

    result_list["error"] = torch.cat(result_list["error"], dim=0)
    result_list["local_diff"] = torch.cat(result_list["local_diff"], dim=0)

    metrics = {
        "error": result_list["error"].mean(0).tolist(),
        "local_diff": result_list["local_diff"].mean(0).tolist(),
        "diff": result_list["diff"].mean(0).tolist(),
        "best_diff": result_list["diff"]
        .reshape(-1, cfg.S)
        .min(1)
        .values.mean(0)
        .tolist(),
        "gt_length": result_list["gt_length"].mean(0).tolist(),
    }
    return metrics


def load_model(ckpt_path, device, load_weight=True):
    cfg_path = ckpt_path.split("/checkpoints")[0] + "/config.yaml"
    cfg = OmegaConf.load(cfg_path)

    class_ = CameraCVAE
    model = class_(cfg)
    if load_weight:
        model.load_state_dict(torch.load(ckpt_path)["state_dict"], strict=False)
    model = model.to(device)
    model.eval()
    return model


def load_seq_list(ds, split, model_cfg):
    if ds == "kitchen_r":
        split = 'all'  # because we don't train on kitchen_r

    DS = lambda seq: RobotData(
        model_cfg,
        split=split,
        train=False,
        mode="all_clip",
        seq_list=[seq],
        fps=0.65,
        dir=f"data/{ds}",
    )
    seq_list = f"data/{ds}/split.json"
    seq_list = sorted(json.load(open(seq_list))[split])    
    return seq_list, DS


@main("config", "eval", version_base=None)
def test(cfg):
    pl.seed_everything(123)
    model = load_model(cfg.ckpt, device)
    model_cfg = model.config
    seq_list, DS = load_seq_list(cfg.ds, cfg.split, model_cfg)

    metric_list = defaultdict(list)
    for seq in tqdm(seq_list):
        dataset = DS(seq)
        val_loader = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=10
        )

        metrics = eval_model(
            model, val_loader, cfg, osp.join(cfg.eval_dir, seq[0].replace("/", "_"))
        )
        metric_list[seq[0]] = metrics

    # save metrics
    os.makedirs(cfg.eval_dir, exist_ok=True)
    with open(osp.join(cfg.eval_dir, f"metrics_{cfg.ds}.json"), "w") as f:
        json.dump(metric_list, f, indent=4)
    return


if __name__ == "__main__":
    test()

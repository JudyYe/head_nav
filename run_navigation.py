# TOOD: initi track first
from glob import glob
import cv2
import time
import imageio
import argparse
from PIL import Image
from tqdm import tqdm
import torch
import numpy as np
from cotracker.predictor import CoTrackerOnlinePredictor

import os
import pybullet as pb
import os.path as osp
import importlib
from collections import defaultdict
import numpy as np
import torch
from hydra import main
from utils import geom_utils
from omegaconf import OmegaConf

from dataset import get_dataloader
from model.network import CameraCVAE
from runtime_modules.camera_module import OpenCVCamera, RedisCamera
from runtime_modules.target_module import TargetServer

# from utils.visualizer import Visualizer

import time
import pickle


import os


##################### Keypoint matching #####################
import cv2
import torch
import numpy as np
from kornia.feature import LoFTR

# choose device and prep LoFTR
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
matcher = LoFTR(pretrained="outdoor").to(device).eval()
print(f"Using device: {device}")


def preprocess_image(image, max_dim=800):
    """Gray, resize, and convert to torch tensor [1x1xHxW]."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # h, w = gray.shape[:2]
    # if max(h, w) > max_dim:
    #     scale = max_dim / max(h, w)
    #     gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(gray).float()[None, None] / 255.0
    return tensor.to(device), gray.shape[::-1]


def find_corresponding_point(img1, img2, pt):
    """Run LoFTR and find the best match to pt (x,y) in img1."""
    t1, _ = preprocess_image(img1)
    t2, _ = preprocess_image(img2)
    with torch.inference_mode():
        matches = matcher({"image0": t1, "image1": t2})
    k0 = matches["keypoints0"].cpu().numpy()
    k1 = matches["keypoints1"].cpu().numpy()
    dists = np.linalg.norm(k0 - np.array(pt), axis=1)
    idx = int(np.argmin(dists))
    if dists[idx] <= 5.0:
        return tuple(k1[idx].astype(int))
    else:
        return None


###############################################################


os.nice(10)
device = "cuda:0"


def load_model(ckpt_path, device, load_weight=True):
    cfg_path = ckpt_path.split("/checkpoints")[0] + "/config.yaml"
    cfg = OmegaConf.load(cfg_path)

    # class_ = getattr(model_module, cfg.MODEL.get("TARGET", "Camera"))
    class_ = CameraCVAE
    model = class_(cfg)
    if load_weight:
        model.load_state_dict(torch.load(ckpt_path)["state_dict"], strict=False)
    model = model.to(device)
    model.eval()
    return model


class ImageManager:
    def __init__(
        self,
    ):
        image_mean = np.array([0.5, 0.5, 0.5])
        image_std = np.array([0.5, 0.5, 0.5])
        self.mean = 255.0 * image_mean
        self.std = 255.0 * image_std

    def init_batch(self, img_list):
        """
        img_list: list of image file path or image array
        """
        videos = []
        for img_file in img_list:
            img = self.preprocess_img(img_file)
            videos.append(img)
        videos = torch.cat(videos, dim=0)  # (T, 3, 256, 256)

        data = {
            "images": videos,
            "camera_prev": torch.zeros(0, 16),
            "goal_2D": torch.zeros(1, 2),
        }

        for k, v in data.items():
            data[k] = v[None].to(device)

        return data

    def update_img(self, batch, videos):
        for img in videos:
            img = self.preprocess_img(img)  # (1, 3, 256, 256)
            batch["images"] = torch.cat([batch["images"][:, 1:], img[None]], dim=1)
        return batch

    def preprocess_img(self, img):
        """
        :param img_file: _description_
        :return: (1, 3, 256, 256)
        """
        img_size = 224
        if isinstance(img, str):
            img = Image.open(img)
        else:
            img = Image.fromarray(img)
        img = img.resize((img_size, img_size), Image.BILINEAR)
        img = np.array(img)

        videos = img[None]  # (1, 256, 256, 3)
        videos = (videos - self.mean[None, None, None, :]) / self.std[
            None, None, None, :
        ]
        videos = torch.FloatTensor(videos).permute(0, 3, 1, 2)
        videos = videos.to(device)
        return videos


class TrackerWrapper:
    def __init__(self, device):
        model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online").to(
            device
        )
        model = model.to(device)
        self.cotracker = model
        self.device = device
        self.window_frames = []
        self.queries = None
        self.is_first_step = True

        self.pred_tracks = None
        self.pred_visibility = None

    def clear_cache(self):
        self.window_frames = []
        self.queries = None
        self.is_first_step = True
        self.pred_tracks = None
        self.pred_visibility = None

    def init_goal(self, goal, img):
        # goal: (B, 2) in [-1, 1]
        # img: (H, W, 3)
        if isinstance(img, str):
            img = imageio.imread(img)
        H, W = img.shape[0:2]
        goal = (goal * 0.5 + 0.5) * torch.tensor(
            [W - 1, H - 1], device=self.device
        )  # (B=1, 2)
        times = torch.zeros_like(goal[..., :1])
        queries = torch.cat([times, goal], dim=-1)  # (B=1, 3)
        self.queries = queries[:, None]  # (B=1, 1, 3)
        # pad ?
        return queries

    def _process_step(self, window_frames, is_first_step, grid_size, grid_query_frame):
        DEFAULT_DEVICE = self.device
        model = self.cotracker
        video_chunk = (
            torch.tensor(
                np.stack(window_frames[-model.step * 2 :]), device=DEFAULT_DEVICE
            )
            .float()
            .permute(0, 3, 1, 2)[None]
        )  # (1, T, 3, H, W)
        return model(
            video_chunk,
            is_first_step=is_first_step,
            grid_size=grid_size,
            grid_query_frame=grid_query_frame,
            queries=self.queries,
            add_support_grid=True,
        )

    def update_goal(self, window_frames, goal):
        pred_tracks, pred_visibility = self._process_step(
            window_frames,
            self.is_first_step,
            grid_size=10,
            grid_query_frame=0,
        )
        self.is_first_step = False

        if pred_tracks is None:
            return goal

        # Update the pred_tracks and pred_visibility in the class
        self.pred_tracks = pred_tracks  # (1, T, P, 2)
        self.pred_visibility = pred_visibility  # (1, T, 1)

        # pred_tracks: (1, T, P, 2)
        goal_2D = pred_tracks[:, -1, :, :]  # (B, P, 2)  [0, H] [0, W]
        H, W = window_frames[0].shape[0:2]
        goal_2D = goal_2D / torch.tensor(
            [W - 1, H - 1], device=self.device
        )  # (B, P, 2)  [-1, 1] [-1, 1]
        goal_2D = goal_2D * 2 - 1  # (B, P, 2)  [-1, 1] [-1, 1]
        goal_2D = goal_2D[:, 0]  # (B, 2)
        return goal_2D


def infer(args):
    vis_dir = "demo"

    os.makedirs(vis_dir, exist_ok=True)
    model = load_model(args.ckpt, device)
    model_cfg = model.config

    print(model_cfg.Tin)

    
    # goal tracker
    tracker = TrackerWrapper(device)
    camera = RedisCamera()
    camera.goal_selector()
    pb.connect(pb.DIRECT)
    vis = TargetServer(visualization=False)
    

    img_list = [camera.get_frame() for _ in range(model_cfg.Tin)]
    # img_list = [imageio.imread(img_file) for img_file in img_list]
    goal_2D = []

    global time_to_switch_to_manip
    global batch

    step = 8
    videos = []


    img_loader = ImageManager()
    batch = img_loader.init_batch(img_list[0 : model_cfg.Tin])

    tracker_init_img = camera.get_zoomed_frame(x=camera.raw_goal[0], 
                                                                y=camera.raw_goal[1],
                                                                i_for_transition=0)
    batch["goal_2D"] = torch.tensor([camera.goal_on_zoomed_img], dtype=torch.float32).to(device)
    tracker.init_goal(batch["goal_2D"], tracker_init_img)


    warmup_goal_canvas_list = []

    warmup_threshold = 100

    for warmup_i in range(warmup_threshold):
        img = camera.get_zoomed_frame(x=camera.raw_goal[0], 
                                      y=camera.raw_goal[1],
                                      i_for_transition=warmup_i,
                                      i_max_for_transition=warmup_threshold,

                                    )
        if len(videos) > 20:
            videos.pop(0)
        videos.append(img)
        
        if warmup_i != 0 and warmup_i % step == 0:
            batch = img_loader.update_img(batch, videos[-model_cfg.Tin :])
            batch["goal_2D"] = tracker.update_goal(videos, batch["goal_2D"])
            goal_canvas, goal_yx_in_pixel_space = vis_goal(batch["goal_2D"], img)
            cv2.imshow("warm up navi Cam Modules", goal_canvas)
            warmup_goal_canvas_list.append(goal_canvas)
            cv2.waitKey(1)
            time.sleep(0.03)
        


    videos = []
    canvas_list = []

    goal_opencv_xy_in_352_352_list = []
    goal_img_in_navi_cam_list = []


    goal_img_in_navi_cam_list.append(img_list[0])
    goal_opencv_xy_in_352_352 = batch["goal_2D"]
    goal_opencv_xy_in_352_352_list.append(
        goal_opencv_xy_in_352_352
    )


    i = 0

    enter_navi_overlap_region_idx = 0
    enter_navi_overlap_region_threshold = 2

    time_to_switch_to_manip = False
    camera.redis_client.set("disable_navi", "false")
    # enable_record = True
    enable_record = False

    recorder = []
    v_cnt = 0


    ts = time.ctime()
    os.mkdir(ts)


    while True:
        time_to_switch_to_manip = (
            enter_navi_overlap_region_idx >= enter_navi_overlap_region_threshold
        )
        if not time_to_switch_to_manip:
            img = camera.get_frame()
            if enable_record:
                recorder.append(img)
                if len(recorder) > 300:
                    np.save(f"{ts}/recordered_video_{v_cnt}.npy", np.array(recorder))
                    v_cnt += 1
                    recorder = []
            if len(videos) > 20:
                videos.pop(0)
            videos.append(img)
            i += 1
            if i != 0 and i % step == 0 and i > 2 * step:
                batch = img_loader.update_img(batch, videos[-model_cfg.Tin :])
                batch["goal_2D"] = tracker.update_goal(videos, batch["goal_2D"])
                

                camera_next, _ = model.sample(
                    batch, z=torch.zeros(1, 1, model_cfg.latent_dim).to(device)
                )
                cTw_pred = model.rollout(camera_next)  # (B, T, 4, 4)
                vis.update(camera_next)  #  cTw_pred)
                # vis cTw
                cam_canvas = vis_cTw(cTw_pred, img)
                # vis goal
                goal_canvas, goal_yx_in_pixel_space = vis_goal(batch["goal_2D"], img)
                # check if goal_2D_in_pixel_space in the perfect overlap region
                object_enter_navi_overlap_region = (
                    camera.check_if_goal_in_overlap_region(goal_yx_in_pixel_space)
                )
                if tracker.pred_visibility is not None:
                    latest_pred_visibility = tracker.pred_visibility[0, -1, 0]
                    print("Visibility: ", latest_pred_visibility,i)
                    # if i % 200 == 0 and (tracker.pred_visibility > 0.5).all():
                    #     tracker.clear_cache()
                    #     tracker.init_goal(batch["goal_2D"], img)
                    #     print("Tracke re-initialized!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    if float(latest_pred_visibility) > 0.5:
                        if object_enter_navi_overlap_region:
                            enter_navi_overlap_region_idx += 1
                            print(
                                f"enter_navi_overlap_region_idx: {enter_navi_overlap_region_idx}"
                            )
                            goal_img_in_navi_cam_list.append(img)
                            goal_opencv_xy_in_352_352 = tracker.pred_tracks[0, -1, 0, :]
                            goal_opencv_xy_in_352_352_list.append(
                                goal_opencv_xy_in_352_352
                            )

                    if (
                        enter_navi_overlap_region_idx
                        == enter_navi_overlap_region_threshold
                    ):
                        # camera.redis_client.set("disable_navi", "true")
                        print("Stop the navigation module")
                        cv2.destroyAllWindows()
                        i = 0
                        canvas_list = []
                        continue
                canvas = np.concatenate([cam_canvas, goal_canvas], axis=1)
                # imageio.imwrite(osp.join(vis_dir, f"{i:04d}.png"), canvas)
                cv2.imshow("Navi Cam Modules", canvas)
                cv2.waitKey(1)
                canvas_list.append(canvas)
                time.sleep(0.03)
        else:
            cv2.destroyAllWindows()
            print(len(goal_opencv_xy_in_352_352_list), goal_opencv_xy_in_352_352_list)
            # for goal_img_in_navi_cam_list and goal_opencv_xy_in_352_352_list, we only keep the last 3 items
            goal_img_in_navi_cam_list = goal_img_in_navi_cam_list
            goal_opencv_xy_in_352_352_list = goal_opencv_xy_in_352_352_list
            assert len(goal_img_in_navi_cam_list) == len(goal_opencv_xy_in_352_352_list)

            valid_dist_list = []

            
            fx, fy, cx, cy = pickle.loads(
                camera.redis_client.get("rs_intrinsics")
            )

            while True:
                disable_navi = camera.redis_client.get("disable_navi").decode()
                if disable_navi == "false":
                    img = camera.get_frame()
                    if enable_record:
                        recorder.append(img)
                        if len(recorder) > 300:
                            np.save(f"recordered_video_{v_cnt}.npy", np.array(recorder))
                            v_cnt += 1
                            recorder = []
                    if len(videos) > 20:
                        videos.pop(0)
                    videos.append(img)
                    i += 1
                    if i != 0 and i % step == 0:
                        batch = img_loader.update_img(batch, videos[-model_cfg.Tin :])
                        batch["goal_2D"] = tracker.update_goal(videos, batch["goal_2D"])

                        camera_next, _ = model.sample(
                            batch, z=torch.zeros(1, 1, model_cfg.latent_dim).to(device)
                        )
                        cTw_pred = model.rollout(camera_next)  # (B, T, 4, 4)
                        vis.update(camera_next)  #  cTw_pred)
                        # vis cTw
                        cam_canvas = vis_cTw(cTw_pred, img)

                        goal_canvas, goal_yx_in_pixel_space = vis_goal(batch["goal_2D"], img)


                        object_enter_navi_overlap_region = (
                            camera.check_if_goal_in_overlap_region(goal_yx_in_pixel_space)
                        )
                        if tracker.pred_visibility is not None:
                            latest_pred_visibility = tracker.pred_visibility[0, -1, 0]
                            print("Visibility: ", latest_pred_visibility,i)
                            # if i % 200 == 0 and (tracker.pred_visibility > 0.5).all():
                            #     tracker.clear_cache()
                            #     tracker.init_goal(batch["goal_2D"], img)
                            #     print("Tracke re-initialized!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                            if float(latest_pred_visibility) > 0.5:
                                if object_enter_navi_overlap_region:
                                    enter_navi_overlap_region_idx += 1
                                    print(
                                        f"enter_navi_overlap_region_idx: {enter_navi_overlap_region_idx}"
                                    )
                                    goal_img_in_navi_cam_list.append(img)
                                    goal_opencv_xy_in_352_352 = tracker.pred_tracks[0, -1, 0, :]
                                    goal_opencv_xy_in_352_352_list.append(
                                        goal_opencv_xy_in_352_352
                                    )

                        canvas = np.concatenate([cam_canvas, goal_canvas], axis=1)
                        # imageio.imwrite(osp.join(vis_dir, f"{i:04d}.png"), canvas)
                        cv2.imshow("[Hope robot to stop] Navi Cam Modules", canvas)
                        cv2.waitKey(1)
                        canvas_list.append(canvas)
                        # time.sleep(0.03)

                time_to_find_corresponding_point = (disable_navi == "true" or i % step == 0)
                if time_to_find_corresponding_point:
                    scale_factor = 0.4
                    # ASSUME THE MANIP CAM IS STILL
                    manip_raw_img, depth = camera.get_manip_cam_frame(scale_factor)
                    depth = pickle.loads(
                                camera.redis_client.get("manip_camera_depth")
                            )

                    # Now, go through the goal_img_in_navi_cam_list and goal_opencv_xy_in_352_352_list
                    matched_point_list = []
                    for idx in range(len(goal_img_in_navi_cam_list)):
                        goal_img_in_navi_cam = goal_img_in_navi_cam_list[idx]
                        goal_opencv_xy_in_352_352 = (
                            goal_opencv_xy_in_352_352_list[idx]
                            .detach()
                            .cpu()
                            .numpy()
                            .reshape(
                                -1,
                            )
                        )
                        # Now, we need to find the corresponding point
                        matched_point = find_corresponding_point(
                            goal_img_in_navi_cam, manip_raw_img, goal_opencv_xy_in_352_352
                        )
                        if matched_point is not None:
                            matched_point_list.append(matched_point)

                    if len(matched_point_list) > 0:
                        # Now, we have the matched_point_list -- compute the median point
                        matched_point_list = np.array(matched_point_list)
                        matched_point_list = matched_point_list[
                            ~np.isnan(matched_point_list).any(axis=1)
                        ]  # remove NaN points
                        if len(matched_point_list) > 0:
                            matched_point_list = matched_point_list.reshape(-1, 2)
                            median_point = np.median(matched_point_list, axis=0)
                            median_point = median_point.astype(np.int32)
                            cv2.circle(
                                manip_raw_img, tuple(median_point), 5, (255, 0, 0), -1
                            )

                            # consider the scale factor to conver the median_point back to the original image
                            median_point = median_point / scale_factor
                            median_point = median_point.astype(np.int32)

                            # get raw depth from manip cam
                            
                            z = depth[median_point[1], median_point[0]]
                            x = (median_point[0] - cx) * z / fx
                            y = (median_point[1] - cy) * z / fy
                            dist = np.sqrt(x**2 + y**2 + z**2)
                            print(f"dist: {dist}", len(valid_dist_list), disable_navi)
                            if dist < 0.85 and dist > 0.2:
                                valid_dist_list.append(dist)
                                if len(valid_dist_list) > 12:
                                    valid_dist_list.pop(0)

                                if len(valid_dist_list) > 3:
                                    camera.redis_client.set("disable_navi", "true")


                                if len(valid_dist_list) > 5:
                                    camera.redis_client.set(
                                        "goal_opencv_xy_in_raw_manip_cam_frame",
                                        pickle.dumps(median_point),
                                    )
                                    cv2.imshow("[sent goal] Manip Cam Modules", manip_raw_img)

                            else:
                                cv2.imshow("[wait...] Manip Cam Modules", manip_raw_img)
                                if len(valid_dist_list) > 0:
                                    valid_dist_list.pop(0)

                            cv2.waitKey(1)
                            time.sleep(0.03)
                    else:
                        print("No matched points found")
                        if len(valid_dist_list) > 0:
                                    valid_dist_list.pop(0)


def vis_cTw(cTw, img):
    T = cTw.shape[1]
    wTc = geom_utils.inverse_rt_v2(cTw)
    H, W = img.shape[0:2]
    img = img.copy()

    wCamera_center = wTc[..., :3, 3]  # (B, T, 3)
    wCamera_center = wCamera_center[0].cpu().detach().numpy()  # (T, 3)
    # wCamera_center[..., 1] = wCamera_center[..., 1] + 1.5
    wCamera_center[..., 2] = wCamera_center[..., 2] + 3
    # print(wCamera_center)

    intr = np.array([[W, 0, W / 2], [0, H, H / 2], [0, 0, 1]])

    iCamera_center = wCamera_center @ intr.T
    iCamera_center = iCamera_center[:, :2] / iCamera_center[:, 2:]  # (T, 2)

    for i in range(1, T):
        x0, y0 = iCamera_center[i - 1]
        x1, y1 = iCamera_center[i]

        x0, y0 = int(x0), int(y0)
        x1, y1 = int(x1), int(y1)

        cv2.line(img, (x0, y0), (x1, y1), (0, 255, 0), 2)

    return img


def vis_goal(goal_2D, img):
    """

    :param goal_2D: (B, 2) in [-1, 1]
    :param img: _description_
    """
    H, W = img.shape[0:2]
    img = img.copy()
    goal_2D = goal_2D[0].cpu().detach().numpy()  # (2, )
    goal_2D = (goal_2D * 0.5 + 0.5) * np.array([W - 1, H - 1])  # (2, )
    goal_2D = goal_2D.astype(np.int32)
    goal_2D_in_pixel_space = goal_2D.copy()
    cv2.circle(img, tuple(goal_2D), 5, (255, 0, 0), -1)
    return img, goal_2D_in_pixel_space


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt", type=str, default="weights/pred/checkpoints/last.ckpt"
    )
    return parser.parse_args()


if __name__ == "__main__":
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)
    args = parse_args()
    # breakpoint()
    infer(args)
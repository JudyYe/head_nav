from copy import deepcopy
import json
import torch
import pickle
import numpy as np
import os.path as osp
from nnutils import geom_utils, mesh_utils
from .dataset import Dataset

from .utils import get_example, get_aug_list


class HumanData(Dataset):
    def __init__(
        self,
        config,
        split="val",
        seq_list=None,
        train=False,
        fps=5,
        preload=True,
        dir="../data",
        mode="rdn_clip",
        **kwargs,
    ):
        self.cfg = config
        self.Tin = config.Tin
        self.Tout = config.Tout
        self.data_dir = dir
        self.clip_dir = osp.join(self.data_dir, "clip")
        self.split = split
        self.train = train
        self.mode = mode

        orig_fps = 10
        target_fps = fps
        self.dt = orig_fps // target_fps

        if seq_list is None:
            seq_list = json.load(open(osp.join(self.data_dir, f"split.json")))[split]

        image_mean = np.array([0.5, 0.5, 0.5])
        image_std = np.array([0.5, 0.5, 0.5])
        self.mean = 255.0 * image_mean
        self.std = 255.0 * image_std
        self.img_size = config.img_size

        self.index_list = self.get_index_list(seq_list)
        self.preload = preload
        if preload:
            self.cache = self.load_cache()

    def load(self, seq):
        if self.preload:
            return deepcopy(self.cache[seq])
        else:
            with open(osp.join(self.clip_dir, seq + ".pkl"), "rb") as f:
                data = pickle.load(f)
            data = self.pad(data, (self.Tin + self.Tout - 1) * self.dt)
            return data

    def load_cache(self):
        print("preloading data")
        # preload all data
        cache = {}
        cache_file = osp.join(self.data_dir, "cache.pkl")
        with open(cache_file, "rb") as fp:
            cache = pickle.load(fp)

        for seq, frame_num in self.index_list:
            data = cache[seq]
            data = self.pad(data, (self.Tin + self.Tout - 1) * self.dt)
            cache[seq] = data
        print("preload done!")
        return cache

    def get_index_list(self, seq_list):
        if self.mode == "rdn_clip":
            return seq_list
        elif self.mode == "all_clip":
            index_list = []
            for seq, frame_num in seq_list:
                for i in range(0, frame_num, self.dt):
                    index_list.append((seq, i, i + (self.Tin + self.Tout) * self.dt))
            return index_list
        else:
            raise NotImplementedError(f"mode {self.mode} not implemented")

    def get_start(self, frame_num):
        return np.random.randint(0, frame_num)

    def pad(self, data, pad_m):
        # repeat last frame
        for k, v in data.items():
            if k in ["video", "wTc"]:
                data[k] = np.concatenate([v, np.repeat(v[-1:], pad_m, axis=0)], axis=0)
        return data

    def get_start_frame(self, idx):
        if self.mode == "rdn_clip":
            seq, frame_num = self.index_list[idx]
            start_ind = self.get_start(frame_num)
            end_ind = start_ind + (self.Tin + self.Tout) * self.dt
        elif self.mode == "all_clip":
            seq, start_ind, end_ind = self.index_list[idx]
        return seq, start_ind, end_ind

    def project_goal(self, wTc_list, intr, goal_world, z_near=0.1):
        """
        :param wTc_list: (T, 4, 4)
        :param intr: (3, 3)
        :param goal_world: (P, 3)
        :return goal_2D: (T, P, 2)
        """
        wTc = wTc_list
        cTw = geom_utils.inverse_rt_v2(wTc)  # (B, 4, 4)
        B = cTw.shape[0]
        goal_world = torch.FloatTensor(goal_world)[None].repeat(B, 1, 1)
        P = goal_world.shape[1]
        goal_cam = mesh_utils.apply_transform(goal_world, cTw)  # (B, P, 3)

        front = (goal_cam[..., 2] > z_near).float()

        intr = torch.FloatTensor(intr)[None, None].repeat(B, P, 1, 1)  # (B, P, 3, 3)
        goal_image = goal_cam[:, :, None] @ intr.transpose(-1, -2)
        goal_image = goal_image.squeeze(-2)
        goal_image = goal_image[:, :, :2] / goal_image[:, :, 2:3]

        goal_image = goal_image * front[..., None]  # (B, P, 2)
        return goal_image, front

    def __getitem__(self, idx):
        seq, start_ind, end_ind = self.get_start_frame(idx)

        data = self.load(seq)
        # subsample here
        for key, value in data.items():
            if key in ["video", "wTc"]:
                data[key] = value[start_ind : end_ind : self.dt]

        videos = np.array(data["video"])
        wTc_list = torch.FloatTensor(np.array(data["wTc"]))
        intr = data["intr"]
        goal_world = data["goal_3D"]

        goal_2D, vis = self.project_goal(wTc_list, intr, goal_world)
        goal_2D = goal_2D.cpu().numpy()
        resized_video = []
        goal_2D_list = []
        aug_list = get_aug_list(self.cfg.DATASETS.CONFIG, len(videos))
        do_flip = aug_list[0][2] and self.train

        if do_flip:
            # flip along x-dim
            flip = torch.FloatTensor(
                [[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
            )
            wTc_list = wTc_list @ flip

        for t, (aug, img) in enumerate(zip(aug_list, videos)):
            center_x = img.shape[1] // 2
            center_y = img.shape[0] // 2
            height = width = max(img.shape[0], img.shape[1])

            (
                img,
                g2d,
                _,
            ) = get_example(
                img,
                center_x,
                center_y,
                width,
                height,
                goal_2D[t],
                self.img_size,
                self.img_size,
                self.mean,
                self.std,
                do_augment=self.train,
                augm_config=None,
                is_bgr=False,
                aug=aug,
            )
            resized_video.append(img)
            goal_2D_list.append(g2d)
        videos = np.stack(resized_video, axis=0)
        goal_2D = np.stack(goal_2D_list, axis=0)

        goal_2D *= 2  # [-1, 1]

        wTc = wTc_list  # [start_ind:end_ind]  # (Tin + Tout + 1, 4, 4)
        wTc1 = wTc[0:-1]  # Tin + Tout
        wTc2 = wTc[1:]
        c1Tw = geom_utils.inverse_rt_v2(wTc1)
        c1Tc2 = c1Tw @ wTc2  # (Tin + Tout - 2, 4, 4)
        camera_prev = c1Tc2[0 : self.Tin - 1]
        camera_next = c1Tc2[self.Tin - 1 :]
        assert len(camera_next) == self.Tout, f"{len(camera_next)} != {self.Tout}"

        videos = videos.astype(np.float32)
        image_prev = videos[0 : self.Tin]
        image_next = videos[self.Tin :]
        T = 200
        if self.train:
            inds = np.random.choice(range(goal_2D.shape[1]), T, replace=True)
            goal_2D = goal_2D[:, inds]
        else:
            goal_2D = goal_2D[:, 0:1]

        rtn = {
            "images": image_prev,
            "camera_prev": camera_prev.reshape(self.Tin - 1, 16),
            "camera_next": camera_next.reshape(self.Tout, 16),
            "images_next": image_next,
            "goal_2D": goal_2D[self.Tin - 1, 0],
            "goal2D_all": goal_2D,
            "is_human": np.array(1, dtype=np.int64),
        }
        return rtn

    def __len__(self):
        return len(self.index_list)

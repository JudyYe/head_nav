import json
import math
import os
import os.path as osp
import pickle
from collections import defaultdict
from glob import glob
from time import time

import cv2
import imageio
import numpy as np
import torch
from fire import Fire
from projectaria_tools.core.sensor_data import TimeDomain
from projectaria_tools.core.sophus import SE3
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from nnutils import geom_utils, image_utils, mesh_utils, plot_utils
from preprocess.utils import AriaData

ratio_4_3 = True


def batch_cvt_data(data_dir="", vis=True, goal_mode="pc"):
    raw_dir = osp.join(data_dir, "raw")
    seq_list = glob(osp.join(raw_dir, "*.vrs"))

    for seq in tqdm(seq_list):
        seq = osp.basename(seq).split(".")[0]
        save_dir = osp.join(osp.dirname(raw_dir), goal_mode)
        cvt_data(seq, goal_mode=goal_mode, save_dir=save_dir, raw_dir=raw_dir, vis=vis)

    zip_all(data_dir=data_dir)

def cvt_data(
    seq="1",
    goal_mode="pc",
    save_dir=None,
    raw_dir=None,
    vis=True,
):
    dataset = AriaData(seq, downsample=1, data_dir=raw_dir)

    clips = get_clip_from_anno(
        dataset,
        osp.join(raw_dir, f"../images/{seq}.json"),
    )

    dataset.load_pc(n_samples=100_000)

    for i, (t_cur, t_next) in enumerate(tqdm(clips)):
        index = f"{seq}_{i:03d}"

        if goal_mode == "pc":
            data = get_goal_by_pc(t_cur, t_next, dataset)
        elif goal_mode == "aug":
            data = aug_view(t_cur, t_next, dataset, desired_theta=60)
            if data is None:
                continue

        os.makedirs(osp.join(save_dir, "clip"), exist_ok=True)
        os.makedirs(osp.join(save_dir, "vis"), exist_ok=True)

        if data["goal_3D"].shape[0] == 0:
            print(f"no goal_3D {index}")
            continue
        with open(osp.join(save_dir, "clip", index + ".pkl"), "wb") as f:
            pickle.dump(data, f)

        # vis data
        if vis:
            vis_data(data, osp.join(save_dir, "vis", index))


def get_goal_by_pc(t_cur, t_next, dataset):
    # vis t_cur to t_next with 10fps
    fps = 10
    video = []
    pc_vis_list = []
    pixel_list = []
    logs = defaultdict(list)
    t = t_cur

    while t < t_next:
        near_gaze = dataset.get_pc_near_gaze_by_time(int((t) * 1e9), nearyby=0.2)  #
        cur_pc_vis, cur_pixel = dataset.visible_pc(int(t * 1e9))  # (P, )
        # if cur_pc_vis is None:
        #     continue
        pc_vis_list.append(cur_pc_vis)
        pixel_list.append(cur_pixel)

        img = dataset.get_image_by_time(int(t * 1e9))
        T_world_device = dataset.get_world_device_by_time(int(t * 1e9))
        T_device_camera = dataset.get_calibration(
            dataset.rgb_camera_calibration
        ).get_transform_device_camera()
        T_world_camera = T_world_device @ T_device_camera
        logs["wTc"].append(T_world_camera.to_matrix())

        video.append(img)
        t += 1 / fps
    pc_vis_list = np.stack(pc_vis_list, axis=0)  # (T, P)

    near_gaze = dataset.get_pc_near_gaze_by_time(
        int((t - 1 / fps) * 1e9), nearyby=0.2
    )  # (P, )
    if near_gaze is None:
        print(f"no near gaze??? {t - t_cur}")
    ind = 0

    # save data
    goal_3D = dataset.pc_world[near_gaze]
    print("goal_3D", goal_3D.shape)
    data = {
        "video": video[ind:],
        "wTc": logs["wTc"][ind:],
        "intr": dataset.get_intr(dataset.downsample),
        "goal_3D": goal_3D,
    }
    return data


def aug_view(i_cur, i_next, dataset: AriaData, desired_theta=60, raw_image=False):
    # vis t_cur to t_next with 10fps
    index_list = dataset.data_provider.get_timestamps_ns(
        dataset.rgb_stream_id, TimeDomain.DEVICE_TIME
    )

    video = []
    logs = defaultdict(list)
    intr = dataset.get_intr(dataset.downsample)

    img = dataset.get_image_by_time(index_list[i_cur])
    W, H = img.shape[1], img.shape[0]
    fov = compute_fov_from_intrinsics(intr, W, H)
    print("ARIA FOV ", fov)
    # Realsense FOV (54.3823304010782, 41.91426799692072)
    # ARIA FOV (98.12651360326811, 98.12651360326811)
    # new ARIA FOV (92.651372980371, 60.98132458579618)

    new_K, new_w, new_h = get_new_camera(intr, gt_new_4_3=ratio_4_3)
    fov = compute_fov_from_intrinsics(new_K, new_w, new_h)
    print("new ARIA FOV", fov)

    orig_img_list = []
    for i in range(i_cur, i_next):
        t = index_list[i]

        img = dataset.get_image_by_time(t)

        T_world_device = dataset.get_world_device_by_time(t)
        T_device_camera = dataset.get_calibration(
            dataset.rgb_camera_calibration
        ).get_transform_device_camera()
        wTc1 = T_world_camera = T_world_device @ T_device_camera

        rot, wTc2 = aug_camera(wTc1, np.deg2rad(desired_theta))

        img = simulate_view(img, intr, new_K, rot, new_w, new_h)

        logs["wTc"].append(wTc2.to_matrix())

        video.append(img)
        if raw_image:
            orig_img = dataset.get_raw_image_by_time(t)
            orig_img_list.append(orig_img)
    # pc_vis_list = np.stack(pc_vis_list, axis=0)  # (T, P)
    near_gaze = dataset.get_pc_near_gaze_by_time(t, nearyby=0.2)
    ind = 0

    # save data
    goal_3D = dataset.pc_world[near_gaze]
    print("goal_3D", goal_3D.shape)
    data = {
        "video": video[ind:],
        "wTc": logs["wTc"][ind:],
        "intr": new_K,
        "goal_3D": goal_3D,
    }
    if raw_image:
        data["orig_img"] = orig_img_list[ind:]
    return data


def get_clip_from_anno(dataset, seq_file):
    clip = []

    with open(seq_file, "r") as f:
        data = json.load(f)
        start_frames = data["start_frames"]
        end_frames = data["end_frames"]
    index_list = dataset.data_provider.get_timestamps_ns(
        dataset.rgb_stream_id, TimeDomain.DEVICE_TIME
    )
    print(f"total index list: {len(index_list)}")
    for start_frame, end_frame in zip(start_frames, end_frames):
        start_time = index_list[start_frame]
        end_time = index_list[end_frame]
        print(f"start_time: {start_time}, end_time: {end_time}")
        clip.append((start_frame, end_frame))
    print(f"total clip: {len(clip)}")
    return clip


def compute_fov_from_intrinsics(K, image_width, image_height):
    f_x = K[0, 0]
    f_y = K[1, 1]

    fov_x = 2 * np.arctan(image_width / (2 * f_x))
    fov_y = 2 * np.arctan(image_height / (2 * f_y))

    # Convert from radians to degrees
    fov_x_deg = np.rad2deg(fov_x)
    fov_y_deg = np.rad2deg(fov_y)

    return fov_x_deg, fov_y_deg


def get_new_camera(intr, gt_new_4_3=False):
    if gt_new_4_3:
        return get_new_camera_43()
    # 1920, 1080 -> ratio = 16:9
    # fov = ???
    scale = 1.1
    intr[0, 0] *= scale
    intr[1, 1] *= scale
    pp_x, pp_y = intr[0][2], intr[1][2]
    # make it to 16:9
    pp_y = pp_x * 9 / 16
    new_K = np.array(
        [[intr[0][0], 0.0, pp_x], [0.0, intr[1][1], pp_y], [0.0, 0.0, 1.0]]
    )
    new_h = int(pp_y * 2)
    new_w = int(pp_x * 2)
    return new_K, new_w, new_h


def get_new_camera_43():
    # Image dimensions
    width = 1920
    height = 1080

    # Diagonal FoV in degrees
    diagonal_fov_deg = 120
    diagonal_fov_rad = math.radians(diagonal_fov_deg)

    # Calculate the diagonal length in pixels
    diagonal_pixels = math.sqrt(width**2 + height**2)

    # Compute focal length from diagonal FoV
    focal_length = (diagonal_pixels / 2) / math.tan(diagonal_fov_rad / 2)

    # fx, fy by computing pixels diagonal - focal_legnth width
    fx = fy = focal_length

    # Assume principal point is at image center
    cx = width / 2
    cy = height / 2

    # Crop width to match 4:3 aspect ratio
    crop_width = int(height * 4 / 3)  # 1080 * 4 / 3 = 1440
    crop_x0 = (width - crop_width) // 2  # center crop
    crop_y0 = 0  # no vertical crop

    # Adjust intrinsics: shift principal point
    cropped_cx = cx - crop_x0
    cropped_cy = cy - crop_y0  # unchanged since no vertical crop

    # New intrinsics after crop
    cropped_intr = np.array([[fx, 0, cropped_cx], [0, fy, cropped_cy], [0, 0, 1]])

    # Target resolution
    target_width = 640
    target_height = 480

    # Scale factors
    scale_x = target_width / crop_width  # 640 / 1440
    scale_y = target_height / height  # 480 / 1080

    # Apply scaling to intrinsics
    final_intr = cropped_intr.copy()
    final_intr[0, 0] *= scale_x  # fx
    final_intr[1, 1] *= scale_y  # fy
    final_intr[0, 2] *= scale_x  # cx
    final_intr[1, 2] *= scale_y  # cy
    return final_intr, target_width, target_height


def aug_camera(wTc1: SE3, desired_theta):
    camera_center_world = wTc1.to_matrix()[:3, 3]

    wZ = np.array([0, 0, -1])
    z_cam = np.array([0, 0, 1])  # camera looking along +Z
    cTw = wTc1.inverse().to_matrix()

    R_cw = cTw[:3, :3]  # 3x3 rotation matrix
    z_world = R_cw.T @ z_cam  # rotate into world frame

    angle_current = angle_of_camera(wTc1)

    axis = np.cross(z_world, wZ)

    wAxis = axis / (np.linalg.norm(axis) + 1e-10)  # normalize
    cAxis = wTc1.inverse().rotation() @ wAxis
    cAxis = cAxis.squeeze(-1)
    axis = cAxis / (np.linalg.norm(cAxis) + 1e-10)  # normalize

    delta_theta = desired_theta - angle_current
    # print("delta theta", np.rad2deg(delta_theta))
    R_aug_world = R.from_rotvec(delta_theta * axis).as_matrix()

    T = np.eye(4)
    T[:3, 3] = -camera_center_world

    T_inv = np.eye(4)
    T_inv[:3, 3] = camera_center_world

    R4 = np.eye(4)
    R4[:3, :3] = R_aug_world

    # Full augmented SE(3) transformation
    R_aug_SE3 = T_inv @ R4 @ T
    R_aug_SE3 = SE3.from_matrix(R_aug_SE3)

    R_aug_SE3 = SE3.from_matrix(R4)

    wTc2 = wTc1 @ R_aug_SE3.inverse()
    # c2Tw = R_aug_SE3 @ cTw

    angle_new = angle_of_camera(wTc2)

    rot = R_aug_SE3.rotation().to_matrix()
    return rot, wTc2


def simulate_view(img, K1, K2, R, new_w, new_h):
    # Compute the homography
    H = K2 @ R @ np.linalg.inv(K1)

    warped_img = cv2.warpPerspective(img, H, (new_w, new_h))

    return warped_img


def angle_of_camera(wTc1: SE3):
    wZ = np.array([0, 0, -1])
    z_cam = np.array([0, 0, 1])  # camera looking along +Z
    cTw = wTc1.inverse().to_matrix()

    R_cw = cTw[:3, :3]  # 3x3 rotation matrix
    z_world = R_cw.T @ z_cam  # rotate into world frame
    cos_theta = np.dot(z_world, wZ) / (np.linalg.norm(z_world) * np.linalg.norm(wZ))
    angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # in radians
    return angle


def vis_data(data, vis_pref):
    # sanity check
    video = data["video"]
    goal_3D = data["goal_3D"]  # (P, 3)
    intr = data["intr"]  # 3,  3
    wTc = data["wTc"]  # (T, 4, 4)

    image_list = []
    for t in range(len(video)):
        img = video[t].copy()

        goal_camera = (SE3.from_matrix(wTc[t]).inverse() @ goal_3D.T).T  # (3, P)
        goal_2D = (intr @ goal_camera.T).T  # (3, P)
        goal_2D = goal_2D[:, :2] / goal_2D[:, 2:3]

        goal_2D = goal_2D.astype(int)

        for x, y in goal_2D:
            if x < 0 or x >= img.shape[1] or y < 0 or y >= img.shape[0]:
                continue
            img = cv2.circle(img, (x, y), 5, (0, 255, 0), -1)

        image_list.append(img)
    imageio.mimsave(vis_pref + "_video.mp4", image_list)

    device = "cuda:0"
    wTc_tensor = torch.FloatTensor(wTc).to(device)
    c1Tw = geom_utils.inverse_rt_v2(wTc_tensor[0:1])
    c1Tc = c1Tw @ wTc_tensor

    mesh_list = plot_utils.vis_cam(wTc=c1Tc, size=0.1)
    coord = plot_utils.create_coord(device, 1, 0.1)

    scene = mesh_utils.join_scene(mesh_list + [coord])
    image_list = mesh_utils.render_geom_rot_v2(scene)
    image_utils.save_gif(image_list, vis_pref + "_cam", ext=".mp4")
    image_utils.save_images(image_list[0], vis_pref + "_cam")


def split(data_dir, val_seq_list=""):
    clip_list = glob(osp.join(data_dir, 'clip', '*.pkl'))
    print(len(clip_list))
    splits = defaultdict(list)
    val_vid = val_seq_list.split(',')

    for clip in clip_list:
        with open(clip, 'rb') as f:
            data = pickle.load(f)
        frame_num = len(data['video'])
        if len(data['goal_3D']) == 0:
            print(f"no goal_3D {clip}")
            continue

        seq = clip.split('/')[-1].split('.')[0]
        if seq.split('_')[0] in val_vid:
            s = 'val'
        else:
            s = 'train'
        splits[s].append((seq, frame_num))
    
    with open(osp.join(data_dir, 'split.json'), 'w') as f:
        json.dump(splits, f, indent=4)
    return 
    

def zip_all(data_dir=""):
    cache = {}
    clip_list = glob(osp.join(data_dir, 'clip', '*.pkl'))
    print(len(clip_list))

    for clip in clip_list:
        with open(clip, 'rb') as f:
            data = pickle.load(f)

        img_size = 256
        H, W = data['video'][0].shape[0:2]
        orig_size = min(H, W)

        scale = orig_size / img_size

        data['intr'][..., 0, :] /= scale
        data['intr'][..., 1, :] /= scale

        new_video = []
        for img in data['video']:
            img = Image.fromarray(img)
            new_w, new_h = int(W / scale), int(H / scale)
            img = img.resize((new_w, new_h))
            new_video.append(np.array(img))
        data['video'] = new_video

        seq = clip.split('/')[-1].split('.')[0]
        cache[seq] = data
    save_file = osp.join(data_dir, 'cache.pkl')
    with open(save_file, 'wb') as fp:
        pickle.dump(cache,  fp)
        print('save to ', save_file)



def main(mode='cvt', data_dir='', val_seq_list='', vis=True, goal_mode='pc'):
    if mode == 'cvt':
        batch_cvt_data(data_dir=data_dir, vis=vis, goal_mode=goal_mode)
    elif mode == 'split':
        split(data_dir=data_dir, val_seq_list=val_seq_list)
    else:
        raise ValueError(f"Invalid mode: {mode}")

if __name__ == "__main__":
    Fire(main)

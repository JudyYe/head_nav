import numpy as np
from skimage.filters import gaussian
import random
import cv2
from typing import List, Tuple
from yacs.config import CfgNode


def get_aug_list(config, num):
    aug = 1.0, 0, False, False, 0, [1.0, 1.0, 1.0], 0., 0.
    if config.get("IID_AUG", 0) == 0:
        aug = do_augmentation(config)
        aug_list = [aug, ] * num
    elif config.get("IID_AUG", 0) == 1:  # all are independent
        aug_list = []
        for i in range(num):
            aug_list.append(do_augmentation(config))
    elif config.get("IID_AUG", 0) == 2:  # only correlate rotation
        aug_list = []
        for i in range(num):
            aug = do_augmentation(config)
            aug = list(aug)
            scale, rot, do_flip, do_extreme_crop, extreme_crop_lvl, color_scale, tx, ty = aug
            if i > 0:
                rot = aug_list[0][1]
            aug_list.append([scale, rot, do_flip, do_extreme_crop, extreme_crop_lvl, color_scale, tx, ty])
    else:
        raise ValueError(f"Unknown IID_AUG {config.get('IID_AUG', 0)}")
    return aug_list


def get_example(img_path: str|np.ndarray, center_x: float, center_y: float,
                width: float, height: float,
                keypoints_2d: np.array, 
                patch_width: int, patch_height: int,
                mean: np.array, std: np.array,
                do_augment: bool, augm_config: CfgNode,
                is_bgr: bool = True,
                use_skimage_antialias: bool = False,
                border_mode: int = cv2.BORDER_CONSTANT,
                return_trans: bool = False,
                other_3dpts_cam: np.array = None,
                return_orig_img: bool = False,
                aug=None,
                ) -> Tuple:
    """
    Get an example from the dataset and (possibly) apply random augmentations.
    Args:
        img_path (str): Image filename
        center_x (float): Bounding box center x coordinate in the original image.
        center_y (float): Bounding box center y coordinate in the original image.
        width (float): Bounding box width.
        height (float): Bounding box height.
        keypoints_2d (np.array): Array with shape (N,3) containing the 2D keypoints in the original image coordinates.
        keypoints_3d (np.array): Array with shape (N,4) containing the 3D keypoints.
        mano_params (Dict): MANO parameter annotations.
        has_mano_params (Dict): Whether MANO annotations are valid.
        flip_kp_permutation (List): Permutation to apply to the keypoints after flipping.
        patch_width (float): Output box width.
        patch_height (float): Output box height.
        mean (np.array): Array of shape (3,) containing the mean for normalizing the input image.
        std (np.array): Array of shape (3,) containing the std for normalizing the input image.
        do_augment (bool): Whether to apply data augmentation or not.
        aug_config (CfgNode): Config containing augmentation parameters.
    Returns:
        return img_patch, keypoints_2d, keypoints_3d, mano_params, has_mano_params, img_size
        img_patch (np.array): Cropped image patch of shape (3, patch_height, patch_height)
        keypoints_2d (np.array): Array with shape (N,3) containing the transformed 2D keypoints.
        keypoints_3d (np.array): Array with shape (N,4) containing the transformed 3D keypoints.
        mano_params (Dict): Transformed MANO parameters.
        has_mano_params (Dict): Valid flag for transformed MANO parameters.
        img_size (np.array): Image size of the original image.
        """
    if isinstance(img_path, str):
        cvimg = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if not isinstance(cvimg, np.ndarray):
            raise IOError("Fail to read %s" % img_path)
    elif isinstance(img_path, np.ndarray):
        cvimg = img_path
    else:
        raise TypeError('img_path must be either a string or a numpy array')
    img_height, img_width, img_channels = cvimg.shape

    img_size = np.array([img_height, img_width])

    if do_augment:
        if aug is not None:
            scale, rot, do_flip, do_extreme_crop, extreme_crop_lvl, color_scale, tx, ty = aug
        else:
            scale, rot, do_flip, do_extreme_crop, extreme_crop_lvl, color_scale, tx, ty = do_augmentation(augm_config)
    else:
        scale, rot, do_flip, do_extreme_crop, extreme_crop_lvl, color_scale, tx, ty = 1.0, 0, False, False, 0, [1.0, 1.0, 1.0], 0., 0.


    if width < 1 or height < 1:
        print(f'{width=}, {height=} break' )
        breakpoint()

    center_x += width * tx  # translation
    center_y += height * ty

    if use_skimage_antialias:
        downsampling_factor = (patch_width / (width*scale))
        if downsampling_factor > 1.1:
            cvimg  = gaussian(cvimg, sigma=(downsampling_factor-1)/2, channel_axis=2, preserve_range=True, truncate=3.0)
    img_patch_cv, trans = generate_image_patch_cv2(cvimg,
                                                    center_x, center_y,
                                                    width, height,
                                                    patch_width, patch_height,
                                                    do_flip, scale, rot, 
                                                    border_mode=border_mode)
    image = img_patch_cv.copy()
    if is_bgr:
        image = image[:, :, ::-1]
    img_patch_cv = image.copy()
    img_patch = convert_cvimg_to_tensor(image)
    
    if return_orig_img:
        cx = cvimg.shape[1] // 2
        cy = cvimg.shape[0] // 2
        HH, WW = cvimg.shape[:2]
        HW = max(HH, WW)
        whole_img_cv, _ = generate_image_patch_cv2(cvimg,
                                                cx, cy, HW, HW, patch_width, patch_height,
                                                do_flip, scale=1, rot=0, border_mode=border_mode)
        if is_bgr:
            whole_img_cv = whole_img_cv[:, :, ::-1]
        whole_img = convert_cvimg_to_tensor(whole_img_cv)

    for n_c in range(min(img_channels, 3)):
        img_patch[n_c, :, :] = np.clip(img_patch[n_c, :, :] * color_scale[n_c], 0, 255)
        if mean is not None and std is not None:
            img_patch[n_c, :, :] = (img_patch[n_c, :, :] - mean[n_c]) / std[n_c]
    if return_orig_img:
        for n_c in range(min(img_channels, 3)):
            whole_img[n_c, :, :] = np.clip(whole_img[n_c, :, :] * color_scale[n_c], 0, 255)
            if mean is not None and std is not None:
                whole_img[n_c, :, :] = (whole_img[n_c, :, :] - mean[n_c]) / std[n_c]
    if do_flip:
        keypoints_2d = fliplr_keypoints(keypoints_2d, img_width,)


    for n_jt in range(len(keypoints_2d)):
        keypoints_2d[n_jt, 0:2] = trans_point2d(keypoints_2d[n_jt, 0:2], trans)
    keypoints_2d = keypoints_2d / patch_width - 0.5

    rtn = [img_patch, keypoints_2d, img_size]
    if return_trans:
        rtn.append(trans)
    if return_orig_img:
        rtn.append(whole_img)
    return rtn



def generate_image_patch_cv2(img: np.array, c_x: float, c_y: float,
                             bb_width: float, bb_height: float,
                             patch_width: float, patch_height: float,
                             do_flip: bool, scale: float, rot: float,
                             border_mode=cv2.BORDER_CONSTANT, border_value=0) -> Tuple[np.array, np.array]:
    """
    Crop the input image and return the crop and the corresponding transformation matrix.
    Args:
        img (np.array): Input image of shape (H, W, 3)
        c_x (float): Bounding box center x coordinate in the original image.
        c_y (float): Bounding box center y coordinate in the original image.
        bb_width (float): Bounding box width.
        bb_height (float): Bounding box height.
        patch_width (float): Output box width.
        patch_height (float): Output box height.
        do_flip (bool): Whether to flip image or not.
        scale (float): Rescaling factor for the bounding box (augmentation).
        rot (float): Random rotation applied to the box.
    Returns:
        img_patch (np.array): Cropped image patch of shape (patch_height, patch_height, 3)
        trans (np.array): Transformation matrix.
    """

    img_height, img_width, img_channels = img.shape
    if do_flip:
        img = img[:, ::-1, :]
        c_x = img_width - c_x - 1


    trans = gen_trans_from_patch_cv(c_x, c_y, bb_width, bb_height, patch_width, patch_height, scale, rot)

    img_patch = cv2.warpAffine(img, trans, (int(patch_width), int(patch_height)), 
                        flags=cv2.INTER_LINEAR, 
                        borderMode=border_mode,
                        borderValue=border_value,
                )
    if (img.shape[2] == 4) and (border_mode != cv2.BORDER_CONSTANT):
        img_patch[:,:,3] = cv2.warpAffine(img[:,:,3], trans, (int(patch_width), int(patch_height)), 
                                            flags=cv2.INTER_LINEAR, 
                                            borderMode=cv2.BORDER_CONSTANT,
                            )

    return img_patch, trans



def gen_trans_from_patch_cv(c_x: float, c_y: float,
                            src_width: float, src_height: float,
                            dst_width: float, dst_height: float,
                            scale: float, rot: float) -> np.array:
    """
    Create transformation matrix for the bounding box crop.
    Args:
        c_x (float): Bounding box center x coordinate in the original image.
        c_y (float): Bounding box center y coordinate in the original image.
        src_width (float): Bounding box width.
        src_height (float): Bounding box height.
        dst_width (float): Output box width.
        dst_height (float): Output box height.
        scale (float): Rescaling factor for the bounding box (augmentation).
        rot (float): Random rotation applied to the box.
    Returns:
        trans (np.array): Target geometric transformation.
    """
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.zeros(2)
    src_center[0] = c_x
    src_center[1] = c_y
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir

    trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans




def convert_cvimg_to_tensor(cvimg: np.array):
    """
    Convert image from HWC to CHW format.
    Args:
        cvimg (np.array): Image of shape (H, W, 3) as loaded by OpenCV.
    Returns:
        np.array: Output image of shape (3, H, W).
    """
    img = cvimg.copy()
    img = np.transpose(img, (2, 0, 1))
    img = img.astype(np.float32)
    return img



def rotate_2d(pt_2d: np.array, rot_rad: float) -> np.array:
    """
    Rotate a 2D point on the x-y plane.
    Args:
        pt_2d (np.array): Input 2D point with shape (2,).
        rot_rad (float): Rotation angle
    Returns:
        np.array: Rotated 2D point.
    """
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)


def do_augmentation(aug_config: CfgNode) -> Tuple:
    """
    Compute random augmentation parameters.
    Args:
        aug_config (CfgNode): Config containing augmentation parameters.
    Returns:
        scale (float): Box rescaling factor.
        rot (float): Random image rotation.
        do_flip (bool): Whether to flip image or not.
        do_extreme_crop (bool): Whether to apply extreme cropping (as proposed in EFT).
        color_scale (List): Color rescaling factor
        tx (float): Random translation along the x axis.
        ty (float): Random translation along the y axis. 
    """

    if random.random() <= aug_config.TRANS_AUG_RATE:
        tx = np.clip(np.random.randn(), -1.0, 1.0) * aug_config.TRANS_FACTOR
        ty = np.clip(np.random.randn(), -1.0, 1.0) * aug_config.TRANS_FACTOR
    else:
        tx = 0
        ty = 0
    
    if random.random() <= aug_config.SCALE_AUG_RATE:    
        scale = np.clip(np.random.randn(), -1.0, 1.0) * aug_config.SCALE_FACTOR + 1.0
    else:
        scale = 1.0
    rot = np.clip(np.random.randn(), -2.0,
                  2.0) * aug_config.ROT_FACTOR if random.random() <= aug_config.ROT_AUG_RATE else 0
    do_flip = aug_config.DO_FLIP and random.random() <= aug_config.FLIP_AUG_RATE
    do_extreme_crop = random.random() <= aug_config.EXTREME_CROP_AUG_RATE
    extreme_crop_lvl = aug_config.get('EXTREME_CROP_AUG_LEVEL', 0)
    c_up = 1.0 + aug_config.COLOR_SCALE
    c_low = 1.0 - aug_config.COLOR_SCALE
    color_scale = [random.uniform(c_low, c_up), random.uniform(c_low, c_up), random.uniform(c_low, c_up)]
    return scale, rot, do_flip, do_extreme_crop, extreme_crop_lvl, color_scale, tx, ty




def fliplr_keypoints(joints: np.array, width: float, flip_permutation: List[int]=None) -> np.array:
    """
    Flip 2D or 3D keypoints.
    Args:
        joints (np.array): Array of shape (N, 3) or (N, 4) containing 2D or 3D keypoint locations and confidence.
        flip_permutation (List): Permutation to apply after flipping.
    Returns:
        np.array: Flipped 2D or 3D keypoints with shape (N, 3) or (N, 4) respectively.
    """
    joints = joints.copy()
    joints[:, 0] = width - joints[:, 0] - 1

    return joints



def trans_point2d(pt_2d: np.array, trans: np.array):
    """
    Transform a 2D point using translation matrix trans.
    Args:
        pt_2d (np.array): Input 2D point with shape (2,).
        trans (np.array): Transformation matrix.
    Returns:
        np.array: Transformed 2D point.
    """
    src_pt = np.array([pt_2d[0], pt_2d[1], 1.]).T
    dst_pt = np.dot(trans, src_pt)
    return dst_pt[0:2]    
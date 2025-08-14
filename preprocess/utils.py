import imageio
import numpy as np
import os
from glob import glob
from tqdm import tqdm
import os.path as osp
from projectaria_tools.core import calibration, data_provider, mps
from projectaria_tools.core.mps.utils import (
    filter_points_from_confidence,
    filter_points_from_count,
    get_gaze_vector_reprojection,
    get_nearest_eye_gaze,
    get_nearest_pose,
    get_nearest_wrist_and_palm_pose,
)
from projectaria_tools.core.sensor_data import SensorData, SensorDataType, TimeDomain
from projectaria_tools.core.sophus import SE3
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.utils.rerun_helpers import AriaGlassesOutline, ToTransform3D
from projectaria_tools.utils.rerun_viewer_mps import (
    get_camera_projection_from_device_point,
)


class AriaData:
    def __init__(self, index="1", downsample=4, data_dir="raw"):
        super().__init__()
        self.data_dir = data_dir
        vrs_path = osp.join(self.data_dir, f"{index}.vrs")
        self.data_provider = data_provider.create_vrs_data_provider(vrs_path)
        self.trajectory_data = mps.read_closed_loop_trajectory(
            str(
                osp.join(
                    self.data_dir, f"mps_{index}_vrs/slam/closed_loop_trajectory.csv"
                )
            )
        )
        self.rgb_stream_id = StreamId("214-1")
        self.rgb_stream_label = self.data_provider.get_label_from_stream_id(
            self.rgb_stream_id
        )

        self.downsample = downsample
        self.pc = None
        self.pc_world = None
        self.pc_path = osp.join(
            self.data_dir, f"mps_{index}_vrs/slam/semidense_points.csv.gz"
        )

        # if personalized_eye_gaze.csv exists, use it else use general_eye_gaze.csv
        eye_gaze_path = osp.join(
            self.data_dir, f"mps_{index}_vrs/eye_gaze/general_eye_gaze.csv"
        )
        self.eyegaze = mps.read_eyegaze(eye_gaze_path)

        device_calibration = self.data_provider.get_device_calibration()
        self.rgb_camera_calibration = device_calibration.get_camera_calib(
            self.rgb_stream_label
        )
        self.T_device_CPF = device_calibration.get_transform_device_cpf()

        self.rgb_linear_camera_calibration = calibration.get_linear_camera_calibration(
            int(self.rgb_camera_calibration.get_image_size()[0]),
            int(self.rgb_camera_calibration.get_image_size()[1]),
            self.rgb_camera_calibration.get_focal_lengths()[0],
            "pinhole",
            self.rgb_camera_calibration.get_transform_device_camera(),
        )
        print(
            "size",
            self.rgb_camera_calibration.get_image_size(),
            self.rgb_linear_camera_calibration,
            self.rgb_camera_calibration,
        )

    def get_pc_near_gaze_by_time(self, timestamp, nearyby=0.2):
        eye_gaze = get_nearest_eye_gaze(self.eyegaze, timestamp)
        if eye_gaze is None:
            return None

        depth_m = eye_gaze.depth or 1.0
        gaze_vector_in_cpf = mps.get_eyegaze_point_at_depth(
            eye_gaze.yaw, eye_gaze.pitch, depth_m
        )
        point_world = self.pc_world

        T_world_device = self.get_world_device_by_time(timestamp)
        T_device_world = T_world_device.inverse()

        origin_device = self.T_device_CPF @ [0, 0, 0]
        vector_device = self.T_device_CPF @ gaze_vector_in_cpf

        origin_world = T_world_device @ origin_device
        vector_world = T_world_device @ vector_device
        direction = (vector_world - origin_world) / np.linalg.norm(
            vector_world - origin_world
        )  # (3, 1)

        distance = point_to_ray_distance(
            point_world, origin_world[:, 0], direction[:, 0]
        )
        device_points = T_device_world @ point_world.T
        device_points = device_points.T
        front_z = device_points[:, 2].squeeze() > 0

        # point_world to origin_world distance
        p2p_distance = np.linalg.norm(
            point_world - origin_world.T, axis=1
        )  # (N, 3) -> (N, )

        near_gaze = (distance < nearyby) & front_z & (p2p_distance < 1.5)
        return near_gaze

    def visible_pc(self, timestamp, batch=True):
        T_world_device = self.get_world_device_by_time(timestamp)
        T_device_world = T_world_device.inverse()
        if T_world_device is None:
            return None, None
        device_points = (T_device_world @ self.pc_world.T).T

        if not batch:
            front_z = np.array(device_points)[:, 2] > 0

            camera_calibration = self.get_calibration(self.rgb_camera_calibration)
            pixels = [
                get_camera_projection_from_device_point(p, camera_calibration)
                for p in device_points
            ]
            visible = [p is not None for p in pixels]
            visible = np.array(visible) & front_z
            for t in range(len(visible)):
                p = pixels[t]
                if p is not None:
                    pixels[t] /= self.downsample
        else:
            camera_calibration = self.get_calibration(self.rgb_camera_calibration)

            pixels, visible = batch_pinhole_project(camera_calibration, device_points)
            pixels = pixels / self.downsample

        return visible, pixels

    def load_pc(self, n_samples=20_000):
        if self.pc is None:
            pc = mps.read_global_point_cloud(self.pc_path)

            # filter the point cloud using thresholds on the inverse depth and distance standard deviation
            inverse_distance_std_threshold = 0.001
            distance_std_threshold = 0.15
            pc = filter_points_from_confidence(
                pc, inverse_distance_std_threshold, distance_std_threshold
            )

            if n_samples > 0:
                pc = filter_points_from_count(pc, n_samples)
            self.pc = pc
            self.pc_world = np.array([point.position_world for point in pc])

        return self.pc

    def post_process_image(
        self, x, rgb_linear_camera_calibration, rgb_camera_calibration
    ):
        img = calibration.distort_by_calibration(
            x,
            rgb_linear_camera_calibration,
            rgb_camera_calibration,
        )
        img = np.rot90(img, k=3)
        return img

    def get_world_device_by_time(self, timestamp):
        pose_info = get_nearest_pose(self.trajectory_data, timestamp)
        if pose_info is None:
            return None
        T_world_device = pose_info.transform_world_device
        assert isinstance(T_world_device, SE3)
        return T_world_device

    def get_depth_by_time(self, timestamp):
        depth_data = (
            self.data_provider.get_depth_image_by_timestamp_ns(
                self.rgb_stream_id,
                timestamp,
                TimeDomain.DEVICE_TIME,
            )
            .data()
            .to_numpy_array()
        )

        return depth_data

    def get_raw_image_by_time(self, timestamp):
        image_data = self.data_provider.get_image_data_by_time_ns(
            self.rgb_stream_id,
            timestamp,
            TimeDomain.DEVICE_TIME,
        )
        data = image_data[0]
        down_sampling_factor = self.downsample
        if hasattr(data, "image_data_and_record"):
            img = data.image_data_and_record()[0].to_numpy_array()
        else:
            img = data.to_numpy_array()
        img = np.rot90(img, k=3)
        if down_sampling_factor > 1:
            img = img[::down_sampling_factor, ::down_sampling_factor]
        return img

    def get_image_by_time(self, timestamp):
        image_data = self.data_provider.get_image_data_by_time_ns(
            self.rgb_stream_id,
            timestamp,
            TimeDomain.DEVICE_TIME,
        )
        data = image_data[0]
        down_sampling_factor = self.downsample
        if hasattr(data, "image_data_and_record"):
            img = data.image_data_and_record()[0].to_numpy_array()
        else:
            img = data.to_numpy_array()

        img = self.post_process_image(
            img, self.rgb_linear_camera_calibration, self.rgb_camera_calibration
        )
        if down_sampling_factor > 1:
            img = img[::down_sampling_factor, ::down_sampling_factor]
        return img

    def get_calibration(
        self,
        rgb_camera_calibration,
        should_rectify_image=True,
        should_rotate_image=True,
    ):
        """
        get intrinsics of pinhole camera, (and T_device_camera from get_transform_device_camera())

        devcie_points -> pixels: get_camera_projection_from_device_point(p, camera_calibration)
        :param rgb_camera_calibration: _description_
        :param should_rectify_image: _description_, defaults to True
        :param should_rotate_image: _description_, defaults to True
        :raises NotImplementedError: _description_
        :return: _description_
        """
        if should_rectify_image:
            rgb_linear_camera_calibration = calibration.get_linear_camera_calibration(
                int(rgb_camera_calibration.get_image_size()[0]),
                int(rgb_camera_calibration.get_image_size()[1]),
                rgb_camera_calibration.get_focal_lengths()[0],
                "pinhole",
                rgb_camera_calibration.get_transform_device_camera(),
            )
            if should_rotate_image:
                rgb_rotated_linear_camera_calibration = (
                    calibration.rotate_camera_calib_cw90deg(
                        rgb_linear_camera_calibration
                    )
                )
                camera_calibration = rgb_rotated_linear_camera_calibration
            else:
                camera_calibration = rgb_linear_camera_calibration
        else:  # No rectification
            if should_rotate_image:
                raise NotImplementedError(
                    "Showing upright-rotated image without rectification is not currently supported.\n"
                    "Please use --no_rotate_image_upright and --no_rectify_image together."
                )
            else:
                camera_calibration = rgb_camera_calibration
        return camera_calibration

    def get_intr(self, down_sampling_factor=1):
        calib = self.get_calibration(self.rgb_camera_calibration)
        intr = np.array(
            [
                [calib.get_focal_lengths()[0], 0, calib.get_principal_point()[0]],
                [0, calib.get_focal_lengths()[1], calib.get_principal_point()[1]],
                [0, 0, 1],
            ]
        )
        intr[0] /= down_sampling_factor
        intr[1] /= down_sampling_factor
        return intr


def point_to_ray_distance(P, O, D):
    """
    Compute the distance from a point P to a ray defined by origin O and direction D.

    Parameters:
        P: (N, 3) array representing N points in 3D space.
        O: (3,) array representing the ray origin.
        D: (3,) array representing the ray direction (should be normalized).

    Returns:
        distances: (N,) array containing the distances of each point to the ray.
    """
    # Ensure direction is normalized
    D = D / np.linalg.norm(D)

    # Vector from origin to point
    OP = P - O

    # Projection of OP onto D
    projection = np.dot(OP, D)[:, None] * D  # (N, 3)

    # Perpendicular vector from P to the ray
    perp_vector = OP - projection

    # Compute distances
    distances = np.linalg.norm(perp_vector, axis=1)

    return distances


def batch_pinhole_project(cali, points, z_near=0.1):
    T_device_camera = cali.get_transform_device_camera()
    points = (T_device_camera.inverse() @ points.T).T
    w, h = cali.get_image_size()
    intr = np.array(
        [
            [cali.get_focal_lengths()[0], 0, cali.get_principal_point()[0]],
            [0, cali.get_focal_lengths()[1], cali.get_principal_point()[1]],
            [0, 0, 1],
        ]
    )  # 3x3
    iPoints = points @ intr.T
    iPoints = iPoints[:, 0:2] / iPoints[:, 2:3]
    visible = (
        (iPoints[:, 0] >= 0)
        & (iPoints[:, 0] < w)
        & (iPoints[:, 1] >= 0)
        & (iPoints[:, 1] < h)
        & (points[:, 2] > z_near)
    )
    return iPoints, visible


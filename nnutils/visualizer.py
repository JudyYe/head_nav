import cv2
from nnutils import mesh_utils, plot_utils, geom_utils
import numpy as np
import torch
from pytorch3d.renderer.cameras import PerspectiveCameras, look_at_view_transform

from . import cam_utils


# visualizer
class Visualizer(object):
    def __init__(self) -> None:
        super().__init__()
        self.color_list = [
            "red",
            "blue",
            "pink",
            "white",
            "yellow",
        ]

    def overlay_one_sample(self, images, goal2D, c1Tc2_list, color_ind=0, H=256, W=256):
        """

        :param images: (B, C, H, W?)
        :param c1Tc2_list: _description_
        :param color_ind: _description_, defaults to 0
        """
        assert images.shape[0] == 1
        device = "cuda:0"

        B = c1Tc2_list[0].shape[0]
        mesh_list = []
        for c, c1Tc2 in enumerate(c1Tc2_list):
            ind = c + color_ind
            color = self.color_list[ind % len(self.color_list)]
            mesh_list.append(self.vis_c1Tc2(c1Tc2, color))

        wCoord = plot_utils.create_coord(device=device, N=B, size=0.1)
        wScene = mesh_utils.join_scene(mesh_list + [wCoord])

        radius = 1
        center = torch.FloatTensor([0, 0, radius]).to(device)[None]  # (1, 3)
        cTw, cameras = get_lookat_cameras(
            None, center=center, max_size=radius * 2, az=180, dist=0.7
        )
        cScene = mesh_utils.apply_transform(wScene, cTw)

        # image_list = mesh_utils.render_geom_rot_v2(cScene)
        image_list = []
        out = mesh_utils.render_mesh(cScene, cameras, out_size=(H, W))

        # rotation y by 90deg
        sTw = torch.FloatTensor(
            [
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [-1, 0, 0, 0],
                [0, 0, 0, 1],
            ]
        ).to(device)[None]
        sScene = mesh_utils.apply_transform(wScene, sTw)
        cScene_side = mesh_utils.apply_transform(sScene, cTw)
        out_side = mesh_utils.render_mesh(cScene_side, cameras, out_size=(H, W))

        right = torch.cat([out["image"], out_side["image"]], dim=-2)

        to_img_fn = lambda x: np.clip(
            (x.permute(0, 2, 3, 1).detach().cpu().numpy() / 2 + 0.5) * 255, 0, 255
        )
        right = to_img_fn(right)[0]

        left = to_img_fn(images)[0].copy()
        goal = goal2D[0]
        goal = goal * 0.5 + 0.5
        goal = goal * images.shape[-2]
        goal = goal.int().cpu().numpy()
        cv2.circle(left, tuple(goal), 5, (255, 0, 0), -1)

        left = cv2.resize(left, (2 * H, 2 * W))
        canvas = np.concatenate([left, right], axis=1)
        canvas = canvas.astype(np.uint8)

        return canvas, image_list

    def frame_videos(self, past, future):
        """
        past: (B, Tin, C, H, W)

        :return (B, Tin+Tout, C, H, W)
        """
        image_prev = self.color_code_images(past, (0, 255, 0))
        image_next = self.color_code_images(future, (255, 0, 0))
        video = torch.cat([image_prev, image_next], dim=1)
        return video

    def color_code_images(self, images, color):
        B, T, C, H, W = images.shape
        images = images.reshape(B * T, C, H, W)
        images = images * 0.5 + 0.5
        images = images.cpu().numpy()
        images = np.clip(images.transpose(0, 2, 3, 1) * 255, 0, 255)
        images = images.astype("uint8")
        for i in range(B * T):
            images[i] = cv2.rectangle(images[i].copy(), (0, 0), (W, H), color, 3)
        images = torch.from_numpy(images)
        images = images.permute(0, 3, 1, 2)
        images = images / 255

        images = images.reshape(B, T, C, H, W)
        return images

    def vis_c1Tc2_list(self, c1Tc2_list, color_ind=0, coord=True):
        """
        :param c1Tc2_list: [list of (B, T, 4*4)]
        """
        device = "cuda:0"

        B = c1Tc2_list[0].shape[0]
        mesh_list = []
        for c, c1Tc2 in enumerate(c1Tc2_list):
            ind = c + color_ind
            color = self.color_list[ind % len(self.color_list)]
            mesh_list.append(self.vis_c1Tc2(c1Tc2, color))

        wCoord = plot_utils.create_coord(device=device, N=B, size=0.1)
        scene = mesh_utils.join_scene(mesh_list + [wCoord])

        image_list = mesh_utils.render_geom_rot_v2(scene)
        return image_list, mesh_list

    def vis_c1Tc2(self, c1Tc2, color):
        B, T_1 = c1Tc2.shape[:2]
        c1Tc2 = c1Tc2.reshape(B, -1, 4, 4)
        cTw = cam_utils.rollout_c1Tc2(c1Tc2)
        T = cTw.shape[1]

        verts, faces = plot_utils.vis_cam(
            cTw=cTw.reshape(B * T, 4, 4), color=color, size=0.1, homo=True
        )
        scene = mesh_utils.join_homo_meshes(
            verts.reshape(B, T, -1, 3),
            faces.reshape(B, T, -1, 3),
        ).to(c1Tc2.device)
        scene.textures = mesh_utils.pad_texture(scene, color)
        return scene

    def vis_goals(self, videos, goals):
        """

        :param videos: (B, T, C, H, W)
        :param goals: (B, T, P, 3)
        """
        B, T, P, _ = goals.shape
        C, H, W = videos.shape[-3:]
        canvas = splat_points(goals.reshape(B * T, -1, 2), (B * T, C, H, W))
        videos = videos.cpu() + canvas.reshape(B, T, C, H, W).cpu()
        return videos


def splat_points(points, canvas_size, r=1):
    """points: (-1, 1)"""
    B, P, _ = points.shape  # B: Batch size, P: Number of points
    N, C, H, W = canvas_size  # Canvas dimensions
    assert C == 3, "Canvas should have 3 channels (RGB)"

    # Generate pixel grid (N, H, W, 2) containing (x, y) coordinates
    y_coords, x_coords = torch.meshgrid(
        torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij"
    )
    pixel_coords = torch.stack([x_coords, y_coords], dim=-1).to(
        points.device
    )  # (H, W, 2)

    # Expand to (N, H, W, P, 2) to compute distances to all points
    pixel_coords = pixel_coords[None, :, :, None, :].expand(N, H, W, P, 2)
    points = points[:, None, None, :, :].expand(N, H, W, P, 2)  # (N, H, W, P, 2)

    # Compute squared Euclidean distance
    dist_sq = torch.sum((pixel_coords - points) ** 2, dim=-1)  # (N, H, W, P)

    # Find the minimum distance to any point at each pixel
    min_dist_sq, _ = torch.min(dist_sq, dim=-1)  # (N, H, W)
    min_dist_sq = min_dist_sq * np.sqrt(H * W)  # Normalize by image size

    # Create canvas, initially red (255, 0, 0)
    red = torch.zeros((N, C, H, W), dtype=torch.uint8, device=points.device)
    red[:, 0, :, :] = 1  # Red channel

    # Set pixels inside radius r to white (255, 255, 255)
    mask = min_dist_sq < (r**2)  # (N, H, W), True if inside radius

    canvas = red * mask[:, None, :, :]

    return canvas  # (N, C, H, W)


def get_lookat_cameras(
    geom,
    focal=10,
    device="cuda:0",
    dist=0.75,
    min_max_size=None,
    center=None,
    max_size=None,
    az=0,
):
    if geom is not None:
        points = mesh_utils.get_verts(geom)  # (N, V, 3)
        center = points.mean(1)  # (N, 3, )
        max_size = points.max(1)[0] - points.min(1)[0]  # (N, 3)
        max_size = max_size.max(-1)[0]  # max of x, y, z
    if min_max_size is not None:
        print(max_size)
        max_size = max_size.clamp(min=min_max_size)
    R, T = look_at_view_transform(
        dist=focal * max_size * dist, elev=0, azim=az, at=center, device=device
    )  # world to view
    cTw = geom_utils.rt_to_homo(R.transpose(-1, -2), T)
    cameras = PerspectiveCameras(focal_length=focal).to(device)

    return cTw, cameras

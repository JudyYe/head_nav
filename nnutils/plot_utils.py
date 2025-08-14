# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
import torch
import torch.nn.functional as F
from pytorch3d.transforms import (
    Rotate,
    Scale,
    Transform3d,
    Translate,
    euler_angles_to_matrix,
)
from . import geom_utils, mesh_utils
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex


# ### Primitives Utils ###
def create_cube(device, N=1, align="center"):
    """
    :return: verts: (1, 8, 3) faces: (1, 12, 3)
    """
    cube_verts = torch.tensor(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ],
        dtype=torch.float32,
        device=device,
    )
    if align == "center":
        cube_verts -= 0.5

    # faces corresponding to a unit cube: 12x3
    cube_faces = torch.tensor(
        [
            [0, 1, 2],
            [1, 3, 2],  # left face: 0, 1
            [2, 3, 6],
            [3, 7, 6],  # bottom face: 2, 3
            [0, 2, 6],
            [0, 6, 4],  # front face: 4, 5
            [0, 5, 1],
            [0, 4, 5],  # up face: 6, 7
            [6, 7, 5],
            [6, 5, 4],  # right face: 8, 9
            [1, 7, 3],
            [1, 5, 7],  # back face: 10, 11
        ],
        dtype=torch.int64,
        device=device,
    )  # 12, 3

    return cube_verts.unsqueeze(0).expand(N, 8, 3), cube_faces.unsqueeze(0).expand(
        N, 12, 3
    )


def create_line(x1, x2, width=None):
    """
    :param x1: Tensor in shape of (N?, 3)
    :param x2: Tensor in shape of (N?, 3)
    :return: padded verts (N, Vcube, 3) and faces (N, Fcube, 3) --> WITHOUT offset
    """
    device = x1.device
    N = len(x1)
    norm = torch.linalg.vector_norm(x2 - x1, dim=-1)  # (N, )
    if width is None:
        thin = norm / 25
    else:
        thin = torch.zeros_like(norm) + width
    scale = Scale(norm, thin, thin, device=device)
    translate = Translate((x2 + x1) / 2, device=device)

    e1 = (x2 - x1) / norm[..., None]

    r = F.normalize(torch.randn_like(e1))
    happy = torch.zeros(
        [
            N,
        ],
        device=device,
        dtype=torch.bool,
    )
    for i in range(10):
        rand = F.normalize(torch.randn_like(e1))
        # print(f"{i} ", rand.shape, e1.shape, r.shape, happy.shape)
        # r[~happy] = rand
        r = torch.where(happy.unsqueeze(-1).repeat(1, 3), r, rand)
        happy = torch.linalg.vector_norm(torch.cross(r, e1, -1), dim=-1).abs() > 1e-6
        # also put to 0 if happy is nan
        happy = happy & ~torch.isnan(happy)
        if torch.all(happy):
            break
    if not torch.all(happy):
        print(
            e1,
            r,
        )
        print("!!!! Warning! Cannot find a vector to orthogonize")
    e2 = torch.cross(e1, r)
    e3 = torch.cross(e1, e2)
    rot = Rotate(
        torch.stack([e1, e2, e3], dim=1), device=device
    )  # seems R is the transposed rot / or row-vector
    # X -> scale -> R -> t, align
    transform = scale.compose(rot, translate)

    each_verts, each_faces, num_cube = 8, 12, 1
    verts, faces = create_cube(device, num_cube)  # (num_cube=1, 8, 3)
    verts = (transform.transform_points(verts)).view(
        N, num_cube * each_verts, 3
    )  # (N, 8, 3)

    verts = verts.expand(N, num_cube * each_verts, 3)
    faces = faces.expand(N, num_cube * each_faces, 3)

    return verts, faces


def create_coord(device, N=1, size=1):
    """Meshes of xyz-axis, each is 1unit, in color RGB

    :param device: _description_
    :param N: _description_
    :return: xyz Meshes in batch N.
    """
    if isinstance(device, int):
        device = torch.device(device)
    scale_size = Scale(
        torch.tensor([size, size, size], dtype=torch.float32, device=device)
    )
    scale = Scale(
        torch.tensor(
            [
                [1, 0.05, 0.05],
                [0.05, 1, 0.05],
                [0.05, 0.05, 1],
            ],
            dtype=torch.float32,
            device=device,
        )
    )
    translate = Translate(
        torch.tensor(
            [
                [0.5, 0, 0],
                [0, 0.5, 0],
                [0, 0, 0.5],
            ],
            dtype=torch.float32,
            device=device,
        )
    )
    rot = Rotate(
        euler_angles_to_matrix(
            torch.tensor(
                [
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                ],
                dtype=torch.float32,
                device=device,
            ),
            "XYZ",
        )
    )
    # X -> scale -> R -> t, align
    transform = scale.compose(rot, translate)
    transform = transform.compose(scale_size)

    each_verts, each_faces, num_cube = 8, 12, 3
    verts, faces = create_cube(device, num_cube)
    verts = (transform.transform_points(verts)).view(
        1, num_cube * each_verts, 3
    )  # (3, 8, 3) -> (1, 32, 3)
    offset = (
        torch.arange(0, num_cube, device=device).unsqueeze(-1).unsqueeze(-1)
        * each_verts
    )  # faces offset
    faces = (faces + offset).view(1, num_cube * each_faces, 3)

    verts = verts.expand(N, num_cube * each_verts, 3)
    faces = faces.expand(N, num_cube * each_faces, 3)

    textures = torch.zeros_like(verts).reshape(N, num_cube, each_verts, 3)
    textures[:, 0, :, 0] = 1
    textures[:, 1, :, 1] = 1
    textures[:, 2, :, 2] = 1
    textures = textures.reshape(N, num_cube * each_verts, 3)

    meshes = Meshes(verts, faces, TexturesVertex(textures)).to(device)
    return meshes


# ######## Cameras Utils ########
def create_camera(device, N, size=0.2, focal=1.0, cam_type="+z"):
    """create N cameras, Meshes of xyz-axis, each is 1unit, in color RGB
    return verts and meshes in shape of (N, Vcam, 3)
    :param focal: focal length in shape of (N, )
    """
    lines = [
        [[0, 0, 0], [1, 1, 1]],
        [[0, 0, 0], [-1, 1, 1]],
        [[0, 0, 0], [1, -1, 1]],
        [[0, 0, 0], [-1, -1, 1]],
        [[1, -1, 1], [1, 1, 1]],
        [[1, -1, 1], [-1, -1, 1]],
        [[-1, 1, 1], [-1, -1, 1]],
        [[-1, 1, 1], [1, 1, 1]],
    ]
    L = 8
    (
        each_verts,
        each_faces,
    ) = 8, 12
    lines = torch.FloatTensor(lines).to(device)  # L, 2, 3?
    lines = lines[None].repeat(N, 1, 1, 1)  # (N, L, 2, 3)
    lines[..., -1] *= focal  # (N, L, 2, 3)
    lines = lines.reshape(N * L, 2, 3)  # (NL, 2, 3)
    x1, x2 = lines.split([1, 1], dim=-2)  # (NL, 1, 3)
    verts, faces = create_line(
        x1.squeeze(-2),
        x2.squeeze(-2),
    )  # (NL, Vcube, 3)
    verts *= size
    verts = verts.reshape(N, L, each_verts, 3)
    faces = faces.reshape(N, L, each_faces, 3)

    offset = torch.arange(0, L, device=device).reshape(1, L, 1, 1) * each_verts
    faces = (faces + offset).reshape(N, L * each_faces, 3)  #

    verts = verts.reshape(N, L * each_verts, 3)
    faces = faces.reshape(N, L * each_faces, 3)
    return verts, faces


def vis_cam(
    wTc=None, cTw=None, color="white", cam_type="+z", size=None, focal=1, homo=False
):
    """visualize camera
    return a List of Meshes, each is a camera mesh in world coordinate
    :param wTc: camera coord to world coord in shape of (4, 4), can have scale, defaults to None
    :param cTw: world coord to camera coord in shape of (4, 4), can have scale, defaults to None
    :param color: ['white', 'red', 'blue', 'yellow']
    :param size: float, camera size
    :param focal: intrinsics, float or tensor in shape of (N, )
    :return: List of Meshes, each is a camera mesh, looks at +z (pytorch3d,opencv,vision convention) or -z (openGL,graphics convention)
    """
    if cTw is not None:
        wTc = geom_utils.inverse_rt_v2(mat=cTw, return_mat=True)
    device = wTc.device
    N = len(wTc)
    dist = mesh_utils.get_camera_dist(wTc=wTc)
    if size is None:
        size = dist.max() * 0.05
        print(size)
    cam_verts, cam_faces = create_camera(
        device, N, size=size, focal=focal, cam_type=cam_type
    )  # (N, Vcam, 3)
    wTc = Transform3d(matrix=wTc.transpose(-1, -2))
    wCam_verts = wTc.transform_points(cam_verts)
    if homo:
        return wCam_verts, cam_faces
    mesh_list = []
    for n in range(N):
        m = Meshes([wCam_verts[n]], [cam_faces[n]])
        m.textures = mesh_utils.pad_texture(m, color)
        mesh_list.append(m)
    return mesh_list

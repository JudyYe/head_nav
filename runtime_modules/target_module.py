import torch
from utils import cam_utils
from utils import geom_utils
import numpy as np
import pickle
from scipy.spatial.transform import Rotation
import pybullet as pb
import numpy as np
import time
import redis
def create_primitive_shape(pb, mass, shape, dim, color=(0.6, 0, 0, 1), 
                           collidable=True, init_xyz=(0, 0, 0),
                           init_quat=(0, 0, 0, 1)):
    # shape: p.GEOM_SPHERE or p.GEOM_BOX or p.GEOM_CYLINDER
    # dim: halfExtents (vec3) for box, (radius, length)vec2 for cylinder, (radius) for sphere
    # init_xyz vec3 being initial obj location, init_quat being initial obj orientation
    visual_shape_id = None
    collision_shape_id = -1
    if shape == pb.GEOM_BOX:
        visual_shape_id = pb.createVisualShape(shapeType=shape, halfExtents=dim, rgbaColor=color)
        if collidable:
            collision_shape_id = pb.createCollisionShape(shapeType=shape, halfExtents=dim)
    elif shape == pb.GEOM_CYLINDER:
        visual_shape_id = pb.createVisualShape(shape, dim[0], [1, 1, 1], dim[1], rgbaColor=color)
        if collidable:
            collision_shape_id = pb.createCollisionShape(shape, dim[0], [1, 1, 1], dim[1])
    elif shape == pb.GEOM_SPHERE:
        visual_shape_id = pb.createVisualShape(shape, radius=dim[0], rgbaColor=color)
        if collidable:
            collision_shape_id = pb.createCollisionShape(shape, radius=dim[0])

    sid = pb.createMultiBody(baseMass=mass, baseInertialFramePosition=[0, 0, 0],
                             baseCollisionShapeIndex=collision_shape_id,
                             baseVisualShapeIndex=visual_shape_id,
                             basePosition=init_xyz, baseOrientation=init_quat)
    return sid

class TargetServer:
    def __init__(self, pred_horizon=10, visualization = False):
        self.pred_horizon = pred_horizon
        self.visualization = visualization
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
        if visualization:
            self.heads = []
            for i in range(pred_horizon):
                self.heads.append(pb.loadURDF("resources/head.urdf", useFixedBase=True))

    def update(self, traj):
        camera_next = traj
        cTw = cam_utils.rollout_c1Tc2(camera_next)  # (B, T, 4, 4)
        c1Tp = geom_utils.inverse_rt_v2(cTw)  # (B, T, 4, 4)
        c1Tp = c1Tp[:]

        # gTc1: x -> -y, y -> -z, z -> x
        gTc1 = torch.FloatTensor([
            [0, 0, 1, 0],
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 1]
        ]).unsqueeze(0).repeat(c1Tp.shape[0], 1, 1).to(c1Tp.device)
        gTp = gTc1 @ c1Tp # (B, T, 4, 4)
        pTg = geom_utils.inverse_rt_v2(gTp)  # (B, T, 4, 4)

        gTp = gTp[0]
        pTg = pTg[0]
        R = gTp[..., :3, :3]
        T = gTp[..., :3, 3]

        # R = pTg[..., :3, :3]
        # T = pTg[..., :3, 3] dR2R1 = R2
        length = len(T)
        traj_pose = np.zeros((length, 7))
        for i in range(length):
            pos = T[i].cpu().detach().numpy()
            traj_pose[i,:3] = pos
            quat = Rotation.from_matrix((R[i]@R[0].inverse()).cpu().detach().numpy()).as_quat()
            traj_pose[i,3:] = quat
            if self.visualization:
                pb.resetBasePositionAndOrientation(self.heads[i], pos, quat)

        self.redis_client.set("traj_pose", pickle.dumps(traj_pose))

if __name__ == "__main__":    
    c = pb.connect(pb.GUI)
    vis = TargetServer()
    input()
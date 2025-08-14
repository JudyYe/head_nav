import torch
from . import geom_utils

def rollout_c1Tc2(c1Tc2, init_cTw=None):
    """

    :param c1Tc2: (B, T-1, 4, 4)
    :param init_cTw: (B, 4, 4)
    :return: cTw in shape of (B, T, 4, 4)
    """
    device = c1Tc2.device
    B, F = c1Tc2.shape[:2]
    if init_cTw is None:
        c1Tw = torch.eye(4, device=device)[None].repeat(B, 1, 1)
    else:
        c1Tw = init_cTw
    cTw = [c1Tw]
    c2Tc1 = geom_utils.inverse_rt_v2(mat=c1Tc2, return_mat=True)

    for f in range(F):
        c1Tw = c2Tc1[:, f] @ c1Tw
        cTw.append(c1Tw)
    cTw = torch.stack(cTw, dim=1)
    return cTw


def get_c1Tc2(cTw):
    """

    :param cTw: (T, 4, 4)
    """
    if cTw.ndim == 3:
        c1Tw = cTw[:-1]
        c2Tw = cTw[1:]
    else:
        c1Tw = cTw[:, :-1]
        c2Tw = cTw[:, 1:]
    wTc2 = geom_utils.inverse_rt_v2(mat=c2Tw, return_mat=True)
    c1Tc2 = c1Tw @ wTc2
    return c1Tc2

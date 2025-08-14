import torch
import torch.nn as nn
from nnutils import geom_utils

from .transformer import TransformerEncoder, TransformerDecoder



class CameraDecoder(nn.Module):
    def __init__(self, inp_dim, cond_dim, dim, latent_dim, Tout, depth=4, heads=8, 
                 mlp_dim=1024, use_2_branch=0, **kwargs):
        super().__init__()
        self.Tout = Tout
        self.use_2_branch = use_2_branch            
        self.z_proj = nn.Linear(latent_dim, cond_dim)
        transformer_args = {
            "num_tokens": 1,
            "token_dim": 1,
            "dim": dim,
            "depth": depth,
            "heads": heads,
            "mlp_dim": mlp_dim,
            "context_dim": cond_dim,
        }
        self.transformer = TransformerDecoder(**transformer_args)
        idty = geom_utils.matrix_to_se3(torch.eye(4).unsqueeze(0), rtn_scale=False)
        init_cam = idty.repeat(self.Tout, 1).reshape(1, self.Tout * 9)
        self.register_buffer('init_cam', init_cam)

        self.deccam = nn.Linear(dim, inp_dim)
        nn.init.zeros_(self.deccam.weight)
        nn.init.zeros_(self.deccam.bias)

        if self.use_2_branch == 1:
            self.deccam_human = nn.Linear(dim, inp_dim)
            nn.init.zeros_(self.deccam_human.weight)
            nn.init.zeros_(self.deccam_human.bias)
            self.register_buffer('init_cam_human', init_cam)        

    def forward(self, tokens, z, context, is_human=None):
        z = self.z_proj(z)
        x = torch.cat([z,context], dim=1)
        x = self.transformer(tokens, context=x)

        if self.use_2_branch == 1:
            x_robot = self.deccam(x) 
            x_robot = x_robot + self.init_cam

            assert is_human is not None, f"is_human is None, but use_2_branch is True"
            x_human = self.deccam_human(x)
            x_human = x_human + self.init_cam_human
            T, N, D = x_human.shape
            is_human_exp = is_human[:, None, None].expand(T, N, D)
            # select by is_human 
            x = torch.where(is_human_exp > 0, x_human, x_robot)
        elif self.use_2_branch == 2: # film layer
            style_embedding = self.style_embedding(is_human)
            # print('style', style_embedding.shape, style_embedding)
            x = self.film(x, style_embedding)
            # print('x', x.shape, x)
            x = self.deccam(x)
            x = x + self.init_cam

        else:
            x_robot = self.deccam(x) 
            x_robot = x_robot + self.init_cam
            x = x_robot
            
        x = x.view(-1, self.Tout, 9)
        return x


class TransformerCVAE(nn.Module):
    def __init__(self, x_dim, latent_dim, cond_dim, hidden_dim, config, **kwargs):
        super().__init__()
        self.Tin = config.Tin
        self.Tout = config.Tout
        self.vid_length = self.Tin
        if config.video_arch == 'temporal':
            length = self.vid_length + 1 + self.Tin
        else:
            length = 300
        self.length = length
        self.latent_dim = latent_dim
        self.encoder = Encoder(self.Tout* 4*4, cond_dim, hidden_dim, latent_dim, 
                               num_tokens=length, **config.enc)

        self.decoder = CameraDecoder(
            x_dim, cond_dim, hidden_dim, latent_dim, self.Tout, **config.dec,
        )
        
    def step(self, context, gt, resample=True, is_human=None):
        """_summary_
        :param context: (B, T, D1)  # vid_length + Tin
        :param gt: (B, Tout, D2)
        :return future camera: (B, Tout, D2)
        """
        rtn = {}
        batch_size = context.shape[0]
        mean, log_var = self.encoder(context, gt)
        if resample:
            z = self.reparameterize(mean, log_var)
        else:
            z = mean

        token = torch.zeros(batch_size, 1, 1).to(context.device)

        x = self.decoder(token, z, context, is_human=is_human)  # (B, Tout, D2)
        rtn["6d"] = x
        mat = geom_utils.se3_to_matrix(x, include_scale=False)

        recon_loss, KLD = self.loss_fn(mat, gt, mean, log_var)

        losses = {"recon": recon_loss, "kl": KLD}
        rtn.update({"losses": losses, 'z': z})
        return mat, rtn

    def forward(self, context, z=None, is_human=None):
        if z is None:
            z = torch.randn(context.shape[0], 1, self.latent_dim, device=context.device)
        batch_size = context.shape[0]
        token = torch.zeros(batch_size, 1, 1).to(context.device)
        x = self.decoder(token, z, context, is_human=is_human)  
        rtn = {}
        rtn["6d"] = x
        mat = geom_utils.se3_to_matrix(x, include_scale=False)
        return mat, rtn

    def loss_fn(self, recon_x, x, mean, log_var):
        batch_size = x.shape[0]
        recon_x = recon_x.reshape(batch_size, -1)
        x = x.reshape(batch_size, -1)

        recon_loss = torch.sum((recon_x - x) ** 2, dim=1).mean()
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=1).mean()
        return recon_loss, KLD

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std




class Encoder(nn.Module):
    """transformer-based encoder"""
    def __init__(self, inp_dim, cond_dim, dim, latent_dim, num_tokens, depth=4, heads=8, 
                 mlp_dim=1024, **kwargs):
        super().__init__()
        self.gt_proj = nn.Linear(inp_dim, cond_dim)
        self.transformer = TransformerEncoder(
            num_tokens=num_tokens,
            token_dim=cond_dim, 
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            )
        self.linear_means = nn.Linear(dim, latent_dim)
        self.linear_log_var = nn.Linear(dim, latent_dim)

    def forward(self, context, gt):
        # context: (B, T, D1)
        # gt: (B, Tout, D2)
        batch_size = context.shape[0]
        gt = self.gt_proj(gt.reshape(batch_size, 1, -1))
        x = torch.cat([context, gt], dim=1)
        x = self.transformer(x)  # (B, T, D)
        x = x.mean(1, keepdim=True)  # (B, 1, D)
        mean = self.linear_means(x)
        log_var = self.linear_log_var(x)
        return mean, log_var        


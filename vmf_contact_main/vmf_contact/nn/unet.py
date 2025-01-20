from typing import Union
import logging
import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange
import math
logger = logging.getLogger(__name__)
from typing import List, Tuple, Sequence
from tqdm import tqdm
import torch.nn.functional as F
# from einops.layers.torch import Rearrange
from .output.transforms import quaternion_to_matrix, quaternion_apply, quaternion_invert, normalize_quaternion
from .output.dist import *

class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            # Rearrange('batch channels horizon -> batch channels 1 horizon'),
            nn.GroupNorm(n_groups, out_channels) if n_groups > 0 else nn.Identity(),
            # Rearrange('batch channels 1 horizon -> batch channels horizon'),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)

class MLPBlock(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_dim=256):
        super().__init__()

        self.block = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.block(x)
    
class SinusoidalPositionEmbeddings(torch.nn.Module):
    """
    dim: Output encoder dimension
    max_val: input assumed to be in 0~max_val
    n: The period of each sinusoidal kernel ranges from 2pi~n*2pi
    """
    def __init__(self, dim: int, max_val: Union[float, int] =1., n: Union[float, int] = 10000.):
        super().__init__()
        self.dim = dim
        assert self.dim % 2 == 0, f"dim must be an even number!"
        self.n = float(n)
        self.max_val = float(max_val)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x / self.max_val * self.n # time: 0~10000

        device = x.device
        dtype = x.dtype
        half_dim = self.dim // 2
        embeddings = math.log(self.n) / (half_dim - 1) # Period: 2pi~10000*2pi
        embeddings = torch.exp(torch.arange(half_dim, device=device, dtype=dtype) * -embeddings) # shape: (self.dim/2, )
        embeddings = x[..., None] * embeddings                                      # shape: (*x.shape, self.dim/2) 
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)        # shape: (*x.shape, self.dim)

        return embeddings # (*x.shape, self.dim)
    
class ConditionalResidualMLP(nn.Module):
    def __init__(self, 
            in_dim, 
            out_dim, 
            cond_dim,
            hidden_dim=256,
            cond_predict_scale=False):
        super().__init__()

        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.Mish(),
                nn.Linear(hidden_dim, out_dim),
            ),
            nn.Sequential(
                nn.Linear(out_dim, hidden_dim),
                nn.Mish(),
                nn.Linear(hidden_dim, out_dim),
            ),
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicqs per-channel scale and bias
        cond_channels = out_dim
        if cond_predict_scale:
            cond_channels = out_dim * 2
        self.cond_predict_scale = cond_predict_scale
        self.out_dim = out_dim
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
        )

        self.residual_conv = nn.Linear(in_dim, out_dim) \
            if in_dim != out_dim else nn.Identity()

    def forward(self, x, cond):
        
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        if self.cond_predict_scale:
            scale = embed[:, :self.out_dim]
            bias = embed[:, self.out_dim:]
            out = scale * out + bias
        else:
            out = out + embed
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out
    
class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, 
            in_channels, 
            out_channels, 
            cond_dim,
            kernel_size=3,
            n_groups=8,
            cond_predict_scale=False):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicqs per-channel scale and bias
        cond_channels = out_channels
        if cond_predict_scale:
            cond_channels = out_channels * 2
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            Rearrange('batch t -> batch t 1'),
        )

        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        '''
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        if self.cond_predict_scale:
            embed = embed.reshape(
                embed.shape[0], 2, self.out_channels, 1)
            scale = embed[:,0,...]
            bias = embed[:,1,...]
            out = scale * out + bias
        else:
            out = out + embed
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out

class ConditionalUnet1D(nn.Module):
    
    q_indices: torch.Tensor
    q_factor: torch.Tensor
    
    def __init__(self, 
        input_dim,
        global_cond_dim=None,
        diffusion_step_embed_dim=256,
        down_dims=[256,512,1024],
        # kernel_size=3,
        # n_groups=8,
        cond_predict_scale=False,
        ang_mult=1.0,
        ):
        
        super().__init__()
        self.register_buffer('q_indices', torch.tensor([[1,2,3], [0,3,2], [3,0,1], [2,1,0]], dtype=torch.long), persistent=False)
        self.register_buffer('q_factor', torch.tensor([[-0.5, -0.5, -0.5], [0.5, -0.5, 0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, 0.5]]), persistent=False)
        if isinstance(down_dims, int):
            down_dims = [down_dims] * 3
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]
        self.ang_mult = ang_mult

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPositionEmbeddings(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed
        if global_cond_dim is not None:
            cond_dim += global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualMLP(
                mid_dim, mid_dim, cond_dim=cond_dim,
                # kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale
            ),
            ConditionalResidualMLP(
                mid_dim, mid_dim, cond_dim=cond_dim,
                # kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale
            ),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualMLP(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    # kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                ConditionalResidualMLP(
                    dim_out, dim_out, cond_dim=cond_dim, 
                    # kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualMLP(
                    dim_out*2, dim_in, cond_dim=cond_dim,
                    # kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                ConditionalResidualMLP(
                    dim_in, dim_in, cond_dim=cond_dim,
                    # kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))
        
        final_conv = nn.Sequential(
            MLPBlock(down_dims[0], 3),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

    def forward(self, 
            sample: torch.Tensor, 
            timestep: Union[torch.Tensor, float, int], 
            global_cond=None, 
            **kwargs):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        # sample = einops.rearrange(sample, 'b h t -> b t h')

        if sample.dim() == 2:
            B, _ = sample.shape
        elif sample.dim() == 3:
            B, T, _ = sample.shape
        assert sample.shape[-1] == 4
        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        # timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)

        if global_cond is not None:
            global_feature = torch.cat([
                global_feature, global_cond
            ], axis=-1)
        
        x = quaternion_to_matrix(sample).flatten(-2)
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            # x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            # x = upsample(x)

        x = self.final_conv(x)

        x = x.reshape(B, -1, 3) if sample.dim() == 3 else x

        x = quaternion_apply(quaternion_invert(sample), x)
        return x
    
    def diffuse_T_target(self,
                         T_target: torch.Tensor, 
                         time: torch.Tensor = .03) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        assert T_target.ndim == 2 # and T_target.shape[-1] == 7, f"{T_target.shape}" # (nT, 7)
        # if len(T_target) != 1:
        #     raise NotImplementedError
        # assert x_ref.ndim == 2 and x_ref.shape[-1] == 3, f"{x_ref.shape}" # (n_xref, 7)

        nT = len(T_target)
        if isinstance(time, float):
            time = torch.tensor([time], device=T_target.device)
        # assert time.shape == (nT,)

        ang_mult = float(self.ang_mult)
        
        #if not time.shape == (1,):
        #    raise NotImplementedError

        eps = time / 2 * (float(ang_mult) ** 2)   # Shape: (1,)
        std = torch.sqrt(time)

        # T, delta_T, (gt_ang_score, gt_lin_score), (gt_ang_score_ref, gt_lin_score_ref) = diffuse_isotropic_se3(T0 = T_target, eps=eps, std=std, x_ref=x_ref, double_precision=True)
        T, delta_T, gt_ang_score, gt_ang_score_ref = diffuse_isotropic_so3_batched(T0 = T_target, eps=eps, std=std, double_precision=True)
        # T, delta_T, (gt_ang_score, gt_lin_score), (gt_ang_score_ref, gt_lin_score_ref) = T.squeeze(-2), delta_T.squeeze(-2), (gt_ang_score.squeeze(-2), gt_lin_score.squeeze(-2)), (gt_ang_score_ref.squeeze(-2), gt_lin_score_ref.squeeze(-2))
        # T, delta_T, gt_ang_score, gt_ang_score_ref = T.squeeze(-2), delta_T.squeeze(-2), gt_ang_score.squeeze(-2), gt_ang_score_ref.squeeze(-2)
        # T: (nT, 7) || delta_T: (nT, 7) || gt_*_score_*: (nT, 3) ||
        # Note that nT = n_samples_x_ref * nT_target  ||   nT_target = 1

        time_in = time.repeat(len(T))

        return T, delta_T, time_in, gt_ang_score, gt_ang_score_ref

    def sample(self, 
               q_seed: torch.Tensor,
               global_cond: torch.Tensor,
               diffusion_schedules: List[Union[List[float], Tuple[float, float]]] = [[1., 0.15], [0.15, 0.09]],
               N_steps: List[int] = [200, 200], 
               timesteps: List[float] = [0.04, 0.04],
               temperatures: Union[Union[int, float], Sequence[Union[int, float]]] = 1.0,
               log_t_schedule: bool = True,
               time_exponent_temp: float = 0.5, # Theoretically, this should be zero.
               time_exponent_alpha: float = 0.5, # Most commonly used exponent in image generation is 1.0, but it is too slow in our case.
               ) -> torch.Tensor:
        """
        alpha = timestep * L^2 * (t^time_exponent_alpha)
        T = temperature * (t^time_exponent_temp)
        """

        if isinstance(temperatures, (int, float)):
            temperatures = [float(temperatures) for _ in range(len(diffusion_schedules))]
        
        # ---------------------------------------------------------------------------- #
        # Convert Data Type
        # ---------------------------------------------------------------------------- #
        device = global_cond.device
        dtype = global_cond.dtype
                
        q = q_seed.clone().detach()
        temperatures = torch.tensor(temperatures, device=device, dtype=torch.float64)
        diffusion_schedules = torch.tensor(diffusion_schedules, device=device, dtype=torch.float64)
        

        # ---------------------------------------------------------------------------- #
        # Begin Loop
        # ---------------------------------------------------------------------------- #
        qs = [q.clone().detach()]
        steps = 0
        for n, schedule in enumerate(diffusion_schedules):
            temperature_base = temperatures[n]
            if log_t_schedule:
                t_schedule = torch.logspace(
                    start=torch.log(schedule[0]), 
                    end=torch.log(schedule[1]), 
                    steps=N_steps[n], 
                    base=torch.e, 
                    device=device, 
                ).unsqueeze(-1)
            else:
                t_schedule = torch.linspace(
                    start=schedule[0], 
                    end=schedule[1], 
                    steps=N_steps[n], 
                    device=device, 
                ).unsqueeze(-1)

            # print(f"{self.__class__.__name__}: sampling with (temp_base: {temperature_base} || t_schedule: {schedule.detach().cpu().numpy()})")
            for i in range(len(t_schedule)):
                t = t_schedule[i]
                temperature = temperature_base * torch.pow(t,time_exponent_temp)
                alpha_ang = (self.ang_mult **2) * torch.pow(t,time_exponent_alpha) * timesteps[n]

                with torch.no_grad():
                    ang_score_dimless = self.forward(q, t.repeat(len(q)).type(dtype), global_cond=global_cond)
                    
                ang_score = ang_score_dimless / (self.ang_mult * torch.sqrt(t))
                ang_noise = torch.sqrt(temperature*alpha_ang) * torch.randn_like(ang_score) 
                ang_disp = (alpha_ang/2) * ang_score + ang_noise

                L = q.detach()[...,self.q_indices] * self.q_factor
                dq = torch.einsum('...ij,...j->...i', L, ang_disp)
                q = normalize_quaternion(q + dq)

                # dT = transforms.se3_exp_map(torch.cat([lin_disp, ang_disp], dim=-1))
                # dT = torch.cat([transforms.matrix_to_quaternion(dT[..., :3, :3]), dT[..., :3, 3]], dim=-1)
                # T = transforms.multiply_se3(T, dT)
                steps += 1
                qs.append(q.clone().detach())

        qs.append(q.clone().detach())
        qs = torch.stack(qs, dim=0).detach()

        return qs


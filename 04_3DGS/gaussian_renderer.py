import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import numpy as np

class GaussianRenderer(nn.Module):
    def __init__(self, image_height: int, image_width: int):
        super().__init__()
        self.H = image_height
        self.W = image_width
        
        # Pre-compute pixel coordinates grid
        y, x = torch.meshgrid(
            torch.arange(image_height, dtype=torch.float32),
            torch.arange(image_width, dtype=torch.float32),
            indexing='ij'
        )
        # Shape: (H, W, 2)
        self.register_buffer('pixels', torch.stack([x, y], dim=-1))

    def compute_projection(
        self,
        means3D: torch.Tensor,          # (N, 3)
        covs3d: torch.Tensor,           # (N, 3, 3)
        K: torch.Tensor,                # (3, 3)
        R: torch.Tensor,                # (3, 3)
        t: torch.Tensor                 # (3)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        N = means3D.shape[0]
        
        # 1. Transform points to camera space
        cam_points = means3D @ R.T + t.unsqueeze(0) # (N, 3)
        
        # 2. Get depths before projection for proper sorting and clipping
        depths = cam_points[:, 2].clamp(min=1.)  # (N, )
        
        # 3. Project to screen space using camera intrinsics
        screen_points = cam_points @ K.T  # (N, 3)
        means2D = screen_points[..., :2] / screen_points[..., 2:3] # (N, 2)
        
        # 4. Transform covariance to camera space and then to 2D
        # Compute Jacobian of perspective projection
        J_proj = torch.zeros((N, 2, 3), device=means3D.device)
        fx = K[0, 0]
        fy = K[1, 1]
        x = cam_points[:, 0]
        y = cam_points[:, 1]
        z = cam_points[:, 2].clamp(min=1e-4) # Avoid division by zero
        
        J_proj[:, 0, 0] = fx / z
        J_proj[:, 0, 2] = -fx * x / (z * z)
        J_proj[:, 1, 1] = fy / z
        J_proj[:, 1, 2] = -fy * y / (z * z)
        
        # Transform covariance to camera space: covs_cam = R @ cov3d @ R.T
        R_expand = R.unsqueeze(0)
        covs_cam = torch.matmul(R_expand, torch.matmul(covs3d, R_expand.transpose(-1, -2)))
        
        # Project to 2D
        covs2D = torch.bmm(J_proj, torch.bmm(covs_cam, J_proj.permute(0, 2, 1)))  # (N, 2, 2)
        
        return means2D, covs2D, depths

    def compute_gaussian_values(
        self,
        means2D: torch.Tensor,    # (N, 2)
        covs2D: torch.Tensor,     # (N, 2, 2)
        pixels: torch.Tensor      # (H, W, 2)
    ) -> torch.Tensor:           # (N, H, W)
        N = means2D.shape[0]
        H, W = pixels.shape[:2]
        
        # Compute offset from mean (N, H, W, 2)
        dx = pixels.unsqueeze(0) - means2D.reshape(N, 1, 1, 2)
        
        # Add small epsilon to diagonal for numerical stability
        eps = 1e-4
        covs2D = covs2D + eps * torch.eye(2, device=covs2D.device).unsqueeze(0)
        
        # Compute determinant for normalization (det = ad - bc)
        det = covs2D[:, 0, 0] * covs2D[:, 1, 1] - covs2D[:, 0, 1] * covs2D[:, 1, 0]
        det = torch.clamp(det, min=1e-6)
        
        # Analytical Inverse of 2x2 covariance matrices
        inv_covs2D = torch.zeros_like(covs2D)
        inv_covs2D[:, 0, 0] = covs2D[:, 1, 1] / det
        inv_covs2D[:, 0, 1] = -covs2D[:, 0, 1] / det
        inv_covs2D[:, 1, 0] = -covs2D[:, 1, 0] / det
        inv_covs2D[:, 1, 1] = covs2D[:, 0, 0] / det
        
        # Extract components for vectorization
        vx = dx[..., 0]   # (N, H, W)
        vy = dx[..., 1]   # (N, H, W)
        a = inv_covs2D[:, 0, 0].view(N, 1, 1)
        b = inv_covs2D[:, 0, 1].view(N, 1, 1)
        d = inv_covs2D[:, 1, 1].view(N, 1, 1)
        
        # Compute Mahalanobis distance
        power = -0.5 * (vx * vx * a + 2.0 * vx * vy * b + vy * vy * d)
        
        # Calculate full Gaussian profile with normalization factor
        norm = 1.0 / (2.0 * np.pi * torch.sqrt(det)).view(N, 1, 1)
        gaussian = norm * torch.exp(power)
    
        return gaussian

    def forward(
            self,
            means3D: torch.Tensor,          # (N, 3)
            covs3d: torch.Tensor,           # (N, 3, 3)
            colors: torch.Tensor,           # (N, 3)
            opacities: torch.Tensor,        # (N, 1)
            K: torch.Tensor,                # (3, 3)
            R: torch.Tensor,                # (3, 3)
            t: torch.Tensor                 # (3, 1)
    ) -> torch.Tensor:
        N = means3D.shape[0]
        
        # 1. Project to 2D, means2D: (N, 2), covs2D: (N, 2, 2), depths: (N,)
        means2D, covs2D, depths = self.compute_projection(means3D, covs3d, K, R, t)
        
        # 2. Depth mask
        valid_mask = (depths > 1.) & (depths < 50.0)  # (N,)
        
        # 3. Sort by depth
        indices = torch.argsort(depths, dim=0, descending=False)  # (N, )
        means2D = means2D[indices]      # (N, 2)
        covs2D = covs2D[indices]       # (N, 2, 2)
        colors = colors[ indices]       # (N, 3)
        opacities = opacities[indices] # (N, 1)
        valid_mask = valid_mask[indices] # (N,)
        
        # 4. Compute gaussian values
        gaussian_values = self.compute_gaussian_values(means2D, covs2D, self.pixels)  # (N, H, W)
        
        # 5. Apply valid mask
        gaussian_values = gaussian_values * valid_mask.view(N, 1, 1)  # (N, H, W)
        
        # 6. Alpha composition setup
        alphas = opacities.view(N, 1, 1) * gaussian_values  # (N, H, W)
        colors = colors.view(N, 3, 1, 1).expand(-1, -1, self.H, self.W)  # (N, 3, H, W)
        colors = colors.permute(0, 2, 3, 1)  # (N, H, W, 3)
        
        # 7. Compute weights: alpha blending and transparency accumulation
        # Clamp alpha in [0.0, 0.999] for numerical stability of transparency product
        alphas_clamped = torch.clamp(alphas, min=0.0, max=0.999)
        one_minus_alpha = 1.0 - alphas_clamped
        cumprod = torch.cumprod(one_minus_alpha, dim=0)
        
        # Shift T_i = \prod_{j<i} (1 - \alpha_j) (first element gets transmission 1.0)
        T = torch.cat([torch.ones(1, self.H, self.W, device=alphas.device), cumprod[:-1]], dim=0)
        weights = alphas * T
        
        # 8. Final rendering
        rendered = (weights.unsqueeze(-1) * colors).sum(dim=0)  # (H, W, 3)
        
        return rendered
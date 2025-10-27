import torch
import torch.nn as nn
import math

class SinusoidalPosEmb(nn.Module):
    """Helper module for sinusoidal time embeddings."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class MLPBlock(nn.Module):
    """A simple MLP block with SiLU activation and LayerNorm."""
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.LayerNorm(dim_out),
            nn.SiLU(),
        )
    def forward(self, x):
        return self.layers(x)

class DemandDiffusionModel(nn.Module):
    """
    A conditional diffusion model for generating demand data.
    This model predicts the noise added to the data at a given timestep.
    """
    def __init__(self, condition_dim: int, data_dim: int = 5, time_emb_dim: int = 64, num_steps: int = 1000):
        super().__init__()
        self.data_dim = data_dim
        self.num_steps = num_steps

        # --- Diffusion Parameters ---
        betas = torch.linspace(1e-4, 0.02, num_steps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('posterior_variance', betas * (1. - torch.roll(alphas_cumprod, 1, 0)) / (1. - alphas_cumprod))

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        self.condition_mlp = nn.Sequential(
            MLPBlock(condition_dim, 128),
            MLPBlock(128, time_emb_dim),
        )

        self.denoising_net = nn.Sequential(
            MLPBlock(data_dim + time_emb_dim, 512),
            MLPBlock(512, 1024),
            MLPBlock(1024, 512),
            nn.Linear(512, data_dim)
        )

    def q_sample(self, x_start, t, noise=None):
        """Forward process: add noise to data."""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # t is expected to be 1D (batch_size,). We reshape here for broadcasting.
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def predict_noise(self, x_t, t, condition):
        """Predicts the noise added to x_t at timestep t."""
        time_emb = self.time_mlp(t)
        cond_emb = self.condition_mlp(condition)
        
        combined_emb = time_emb + cond_emb
        
        emb_reshaped = combined_emb.unsqueeze(1)
        emb_expanded = emb_reshaped.expand(-1, x_t.shape[1], -1)
        
        net_input = torch.cat((x_t, emb_expanded), dim=-1)
        
        return self.denoising_net(net_input)

    def forward(self, x_start, condition):
        """
        The training forward pass.
        """
        batch_size, num_demands, _ = x_start.shape
        
        t = torch.randint(0, self.num_steps, (batch_size,), device=x_start.device).long()
        
        noise = torch.randn_like(x_start)
        
        # Pass the 1D tensor 't' directly to q_sample.
        x_t = self.q_sample(x_start, t, noise)
        
        predicted_noise = self.predict_noise(x_t, t, condition)
        
        return noise, predicted_noise

    @torch.no_grad()
    def sample(self, condition: torch.Tensor, num_demands: int, grid_size: tuple[int, int]) -> torch.Tensor:
        """
        The generation/sampling process (DDPM reverse process).
        """
        self.eval()
        device = next(self.parameters()).device
        
        x_t = torch.randn((1, num_demands, self.data_dim), device=device)
        
        for t_int in reversed(range(self.num_steps)):
            t = torch.full((1,), t_int, device=device, dtype=torch.long)
            
            predicted_noise = self.predict_noise(x_t, t, condition)
            
            alpha_t = 1. - self.betas[t]
            alpha_t_cumprod = self.alphas_cumprod[t]
            
            term1 = (1 / torch.sqrt(alpha_t))
            term2 = (self.betas[t] / self.sqrt_one_minus_alphas_cumprod[t])
            
            x_t = term1 * (x_t - term2 * predicted_noise)
            
            if t_int > 0:
                variance = self.posterior_variance[t]
                if variance > 0:
                    noise = torch.randn_like(x_t)
                    x_t += torch.sqrt(variance) * noise
        
        self.train()
        return x_t.squeeze(0)
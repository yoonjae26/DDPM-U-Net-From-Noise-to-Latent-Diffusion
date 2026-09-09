import torch
from latent_diffusion_playground.diffusion.forward import q_sample
from latent_diffusion_playground.diffusion.reverse import ddim_sample_step, p_sample_step, predict_x0_from_epsilon
from latent_diffusion_playground.diffusion.scheduler import DiffusionSchedule, make_schedule

@torch.no_grad()
def p_sample_loop(model, shape, schedule, device):
    """Runs the full reverse diffusion process to generate a sample from noise."""
    model.eval()
    img = torch.randn(shape, device=device)
    for i in reversed(range(schedule.timesteps)):
        t = torch.full((shape[0],), i, device=device, dtype=torch.long)
        epsilon = model(img, t)
        img = p_sample_step(img, t, epsilon, schedule)
    return img

__all__ = [
	"DiffusionSchedule",
	"make_schedule",
	"q_sample",
	"p_sample_step",
	"ddim_sample_step",
	"predict_x0_from_epsilon",
	"p_sample_loop",
]

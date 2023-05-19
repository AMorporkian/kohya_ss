from library.custom_train_functions import apply_noise_offset, pyramid_noise_like
import torch


def noise_prediction(accelerator, unet, encoder_hidden_states, timesteps, noisy_latents):
    with accelerator.autocast():
        noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
    return noise_pred


def get_target_type(args, noise_scheduler, latents, noise, timesteps):
    """Check if we are doing v-parameterization training or not."""
    if args.v_parameterization or args.v_noise:
                    # v-parameterization training
        target = noise_scheduler.get_velocity(latents, noise, timesteps)
    else:
        target = noise
    return target


def get_latents_from_noise(noise_scheduler, latents, b_size, noise):
    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (b_size,), device=latents.device)
    timesteps = timesteps.long()
                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
    return timesteps,noisy_latents


def sample_noise(args, latents):
    noise = torch.randn_like(latents, device=latents.device)
    if args.noise_offset:
        noise = apply_noise_offset(latents, noise, args.noise_offset, args.adaptive_noise_scale)
    elif args.multires_noise_iterations:
        noise = pyramid_noise_like(noise, latents.device, args.multires_noise_iterations, args.multires_noise_discount)
    return noise
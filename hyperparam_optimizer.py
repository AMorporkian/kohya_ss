import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import train_network

wandb.init(project="kohya_splatoon")

sweep_configuration = {
    'method': 'bayes',
    'name': 'sweep',
    'metric': {
        'goal': 'minimize', 
        'name': 'loss_val'
        },
    'parameters': {
        'batch_size': {'min': 1, 'max': 5},
        'num_restarts': {'min': 1, 'max': 10},
        'unet_lr': {'max': 0.005, 'min': 0.00001},
        'text_encoder_lr': {'max': 0.005, 'min': 0.00001},
        'network_alpha': {[1,8,16,24,32,40,48,56,64,72,80,88,96]},
        'epochs': {'min': 1, 'max': 10},
     }
}

args = {'v2': False, 
        'v_parameterization': True, 
        'pretrained_model_name_or_path': 
        '/workspace/src/kohya_ss/anyloraCheckpoint_novaeFp16.ckpt', 
        'tokenizer_cache_dir': None, 
        'train_data_dir': '/workspace/datasets/harmony_splatoon/img', 
        'shuffle_caption': False, 
        'caption_extension': '.caption', 
        'caption_extention': None, 
        'keep_tokens': 0, 
        'color_aug': True, 
        'flip_aug': True, 
        'face_crop_aug_range': None, 
        'random_crop': False, 
        'debug_dataset': False, 
        'resolution': '576', 
        'cache_latents': False, 
        'vae_batch_size': 1, 
        'cache_latents_to_disk': False, 
        'enable_bucket': False, 
        'min_bucket_reso': 256, 
        'max_bucket_reso': 1024, 
        'bucket_reso_steps': 64, 
        'bucket_no_upscale': False, 
        'token_warmup_min': 1, 
        'token_warmup_step': 0, 
        'caption_dropout_rate': 0.0, 
        'caption_dropout_every_n_epochs': 0,
        'caption_tag_dropout_rate': 0.0, 
        'reg_data_dir': None, 
        'in_json': None, 
        'dataset_repeats': 1,
        'output_dir': '/workspace/datasets/harmony_splatoon/out1', 
        'output_name': None,
        'huggingface_repo_id': None, 
        'huggingface_repo_type': None, 
        'huggingface_path_in_repo': None, 
        'huggingface_token': None, 
        'huggingface_repo_visibility': None, 
        'save_state_to_huggingface': False, 
        'resume_from_huggingface': False, 
        'async_upload': False, 
        'save_precision': None, 
        'save_every_n_epochs': 1, 
        'save_every_n_steps': 50, 
        'save_n_epoch_ratio': None, 
        'save_last_n_epochs': None, 
        'save_last_n_epochs_state': None, 
        'save_last_n_steps': None, 
        'save_last_n_steps_state': None, 
        'save_state': False, 
        'resume': None, 
        'train_batch_size': 10, 
        'max_token_length': None, 
        'mem_eff_attn': False, 
        'xformers': True, 
        'vae': '/workspace/automatic/models/VAE/kl-f8-anime2.ckpt', 
        'max_train_steps': None, 
        'max_train_epochs': None, 
        'max_data_loader_n_workers': 8, 
        'persistent_data_loader_workers': False, 
        'seed': 784793, 
        'gradient_checkpointing': False, 
        'gradient_accumulation_steps': 1, 
        'mixed_precision': 'bf16', 
        'full_fp16': False, 
        'clip_skip': 2, 
        'logging_dir': '/workspace/datasets/harmony_splatoon/log', 
        'log_with': 'wandb', 
        'log_prefix': None, 
        'log_tracker_name': "SD_Optimization", 
        'wandb_api_key': None, 
        'noise_offset': 0.1, 
        'multires_noise_iterations': None, 
        'multires_noise_discount': 0.3, 
        'adaptive_noise_scale': 0.01, 
        'lowram': False, 
        'sample_every_n_steps': None, 
        'sample_every_n_epochs': 1, 
        'sample_prompts': '/workspace/datasets/harmony_splatoon/model/sample/prompt.txt', 
        'sample_sampler': 'k_dpm_2', 
        'config_file': None, 
        'output_config': False, 
        'prior_loss_weight': 1.0, 
        'optimizer_type': '', 
        'use_8bit_adam': True, 
        'use_lion_optimizer': False, 
        'learning_rate': 1e-05, 
        'max_grad_norm': 1.0, 
        'optimizer_args': None,
        'lr_scheduler_type': '', 
        'lr_scheduler_args': None, 
        'lr_scheduler': 'cosine_with_restarts', 
        'lr_warmup_steps': 0,
        'lr_scheduler_num_cycles': 3,
        'lr_scheduler_power': 1,
        'dataset_config': None,
        'min_snr_gamma': 1.0, 
        'weighted_captions': False, 
        'no_metadata': False, 
        'v_noise': True, 
        'v_noise_gamma': 1.0, 
        'save_model_as': 'safetensors', 
        'unet_lr': None, 
        'text_encoder_lr': None, 
        'network_weights': None, 
        'network_module': 'lycoris.kohya', 
        'network_dim': 128, 
        'network_alpha': 8.0, 
        'network_args': None, 
        'network_train_unet_only': False, 
        'network_train_text_encoder_only': False, 
        'training_comment': None, 
        'dim_from_weights': False
        }
# tokenizer = , current_epoch, current_step, accelerator, unwrap_model, weight_dtype, text_encoder, vae, unet, network, train_text_encoder, optimizer, train_dataloader, lr_scheduler, metadata, progress_bar, global_step, noise_scheduler, loss_list, loss_total, on_step_start, save_model, remove_model, epoch = args, 

sweep_id = wandb.sweep(sweep_configuration, project="SD_Optimization")

def train_one_epoch(batch_size, num_restarts, unet_lr, text_encoder_lr, network_alpha, epochs):
    a = args.copy()
    a['batch_size'] = batch_size
    a['num_restarts'] = num_restarts
    a['unet_lr'] = unet_lr
    a['text_encoder_lr'] = text_encoder_lr
    a['network_alpha'] = network_alpha
    a['max_train_epochs'] = epochs

    return train_network.train(a)


def main():
    run = wandb.init()

    # note that we define values from `wandb.config`  
    # instead of defining hard values
    bs = wandb.config.batch_size
    num_restarts = wandb.config.num_restarts
    unet_lr = wandb.config.unet_lr
    text_encoder_lr = wandb.config.text_encoder_lr
    network_alpha = wandb.config.network_alpha
    epochs = wandb.config.epochs
    for epoch in np.arange(1, epochs):
      train_loss_, val_loss_ = train_one_epoch(epoch)


      wandb.log({
        'epoch': epoch, 
        'train_loss': train_loss_,
        'loss_val': val_loss_,
      })

# Start sweep job.
wandb.agent(sweep_id, function=main, count=4)


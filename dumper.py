import argparse
import toml
import os

# Define the arguments that we expect
parser = argparse.ArgumentParser()
parser.add_argument("--pretrained_model_name_or_path")
parser.add_argument("--vae")
parser.add_argument("--sample_every_n_steps", type=int)
parser.add_argument("--train_data_dir")
parser.add_argument("--output_dir")
parser.add_argument("--train_batch_size", type=int)
parser.add_argument("--xformers", action='store_true')
parser.add_argument("--mixed_precision")
parser.add_argument("--clip_skip", type=int)
parser.add_argument("--save_every_n_epochs", type=int)
parser.add_argument("--sample_prompts")
parser.add_argument("--log_with")
parser.add_argument("--network_module")
parser.add_argument("--network_dim", type=int)
parser.add_argument("--use_8bit_adam", action='store_true')
parser.add_argument("--logging_dir")
parser.add_argument("--v_noise", action='store_true')
parser.add_argument("--v_parameterization", action='store_true')
parser.add_argument("--v_noise_gamma", type=float)
parser.add_argument("--resolution", type=int)
parser.add_argument("--min_snr_gamma", type=int)
parser.add_argument("--adaptive_noise_scale", type=float)
parser.add_argument("--noise_offset", type=float)
parser.add_argument("--sample_sampler")
parser.add_argument("--color_aug", action='store_true')
parser.add_argument("--flip_aug", action='store_true')
parser.add_argument("--lr_scheduler")
parser.add_argument("--lr_scheduler_num_cycles", type=int)
parser.add_argument("--seed", type=int)
parser.add_argument("--max_train_steps", type=int)
parser.add_argument("--save_every_n_steps", type=int)
parser.add_argument("--network_alpha", type=int)
parser.add_argument("--learning_rate", type=float)

# Parse the arguments
args = parser.parse_args()

# Write the arguments to a TOML file
with open('args.toml', 'w') as f:
    toml.dump(vars(args), f)

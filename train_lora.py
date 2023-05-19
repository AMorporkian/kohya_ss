import argparse
import random


from accelerate.utils import set_seed

import library.train_util as train_util
from setup_parser import setup_parser
from trainer import train


# TODO 他のスクリプトと共通化する
def generate_step_logs(args: argparse.Namespace, current_loss, avr_loss, lr_scheduler):
    logs = {"loss/current": current_loss, "loss/average": avr_loss}

    lrs = lr_scheduler.get_last_lr()

    if args.network_train_text_encoder_only or len(lrs) <= 2:  # not block lr (or single block)
        if args.network_train_unet_only:
            logs["lr/unet"] = float(lrs[0])
        elif args.network_train_text_encoder_only:
            logs["lr/textencoder"] = float(lrs[0])
        else:
            logs["lr/textencoder"] = float(lrs[0])
            logs["lr/unet"] = float(lrs[-1])  # may be same to textencoder

        if args.optimizer_type.lower().startswith("DAdapt".lower()):  # tracking d*lr value of unet.
            logs["lr/d*lr"] = lr_scheduler.optimizers[-1].param_groups[0]["d"] * lr_scheduler.optimizers[-1].param_groups[0]["lr"]
    else:
        idx = 0
        if not args.network_train_unet_only:
            logs["lr/textencoder"] = float(lrs[0])
            idx = 1

        for i in range(idx, len(lrs)):
            logs[f"lr/group{i}"] = float(lrs[i])
            if args.optimizer_type.lower().startswith("DAdapt".lower()):
                logs[f"lr/d*lr/group{i}"] = (
                    lr_scheduler.optimizers[-1].param_groups[i]["d"] * lr_scheduler.optimizers[-1].param_groups[i]["lr"]
                )

    return logs
    
def setup_rng(args):
    if args.seed is None:
        args.seed = random.randint(0, 2**32)
    set_seed(args.seed)

def do_logging(args, accelerator, lr_scheduler, global_step, current_loss, avr_loss):
    if args.logging_dir is not None:
        logs = generate_step_logs(args, current_loss, avr_loss, lr_scheduler)
        accelerator.log(logs, step=global_step)

if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    args = train_util.read_config_from_file(args, parser)
    train(args)









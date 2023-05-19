import gc
import importlib
import sys

import torch
from get_combined_metadata import generate_metadata, get_combined_metadata
import library.config_util as config_util
import library.huggingface_util as huggingface_util
import library.train_util as train_util
from library.config_util import BlueprintGenerator, ConfigSanitizer, get_user_config
from library.custom_train_functions import apply_snr_weight, get_weighted_text_embeddings, patch_scheduler_betas
from library.train_util import save_files
from noise_prediction import get_latents_from_noise, get_target_type, noise_prediction, sample_noise
from train_lora import do_logging, setup_rng


from tqdm import tqdm


import math
import os
import random
import time
from multiprocessing import Value


# def print_generation_information(args, train_dataset_group, train_dataloader, num_train_epochs):
#     print(f"""running training / 学習開始
#           num train images * repeats / 学習画像の数×繰り返し回数: {train_dataset_group.num_train_images})
#           num reg images / 正則化画像の数: {train_dataset_group.num_reg_images}

def print_generation_information(train_dataset_group, train_dataloader, num_train_epochs, max_train_steps, gradient_accumulation_steps):
    print(f"""running training / 学習開始
          num train images * repeats / 学習画像の数×繰り返し回数: {train_dataset_group.num_train_images})
          num reg images / 正則化画像の数: {train_dataset_group.num_reg_images}
          num batches per epoch / 1epochのバッチ数: {len(train_dataloader)}
          num epochs / epoch数: {num_train_epochs}
          batch size per device / バッチサイズ: {', '.join([str(d.batch_size) for d in train_dataset_group.datasets])}
          gradient accumulation steps / 勾配を合計するステップ数 = {gradient_accumulation_steps}
          total optimization steps / 学習ステップ数: {max_train_steps}""")
        #   num batches per epoch / 1epochのバッチ数: {len(train_dataloader)}
        #   num epochs / epoch数: {num_train_epochs}
        #   batch size per device / バッチサイズ: {', '.join([str(d.batch_size) for d in train_dataset_group.datasets])}
        #   gradient accumulation steps / 勾配を合計するステップ数 = {args.gradient_accumulation_steps}
        #   total optimization steps / 学習ステップ数: {args.max_train_steps}""")

def prepare_networks(accelerator, train_unet, train_text_encoder):
    # if train_unet and train_text_encoder:
    #     unet, text_encoder, network, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    #         unet, text_encoder, network, optimizer, train_dataloader, lr_scheduler
    #     )
    # elif train_unet:
    #     unet, network, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    #         unet, network, optimizer, train_dataloader, lr_scheduler
    #     )
    # elif train_text_encoder:
    #     text_encoder, network, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    #         text_encoder, network, optimizer, train_dataloader, lr_scheduler
    #     )
    # else:
    #     network, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(network, optimizer, train_dataloader, lr_scheduler)
    # return text_encoder,unet,network,optimizer,train_dataloader,lr_scheduler

    # A redesign of the above code to make it more readable
    if train_unet and train_text_encoder:
        unet, text_encoder, network, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            train_unet, train_text_encoder, network, optimizer, train_dataloader, lr_scheduler
        )
    elif train_unet:
        unet, network, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            train_unet, network, optimizer, train_dataloader, lr_scheduler
        )
    elif train_text_encoder:
        text_encoder, network, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            train_text_encoder, network, optimizer, train_dataloader, lr_scheduler
        )
    else:
        network, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(network, optimizer, train_dataloader, lr_scheduler)
    return text_encoder,unet,network,optimizer,train_dataloader,lr_scheduler

def prepare_gradient_checkpointing(text_encoder, unet, network):
    unet.enable_gradient_checkpointing()
    text_encoder.gradient_checkpointing_enable()
    network.enable_gradient_checkpointing()


def prepare_lora(args, text_encoder, vae, unet):
    sys.path.append(os.path.dirname(__file__))
    print("import network module:", args.network_module)
def prepare_lr_scheduler(args, network, train_unet, train_text_encoder):
    if train_unet and train_text_encoder:
        lr_scheduler = network.configure_schedulers()
    elif train_unet:
        lr_scheduler = network.configure_schedulers()
    elif train_text_encoder:
        lr_scheduler = network.configure_schedulers()
    else:
        lr_scheduler = network.configure_schedulers()
    return lr_scheduler    network_module = importlib.import_module(args.network_module)

    net_kwargs = {}
    if args.network_args is not None:
        for net_arg in args.network_args:
            key, value = net_arg.split("=")
            net_kwargs[key] = value

    # if a new network is added in future, add if ~ then blocks for each network (;'∀')
    if args.dim_from_weights:
        network, _ = network_module.create_network_from_weights(1, args.network_weights, vae, text_encoder, unet, **net_kwargs)
    else:
        network = network_module.create_network(1.0, args.network_dim, args.network_alpha, vae, text_encoder, unet, **net_kwargs)
    if network is None:
        return

    if hasattr(network, "prepare_network"):
        network.prepare_network(args)
    return net_kwargs,network


def vae_latent_cache(args, train_dataset_group, accelerator, weight_dtype, vae):
    vae.to(accelerator.device, dtype=weight_dtype)
    vae.requires_grad_(False)
    vae.eval()
    with torch.no_grad():
        train_dataset_group.cache_latents(vae, args.vae_batch_size, args.cache_latents_to_disk, accelerator.is_main_process)
    vae.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    accelerator.wait_for_everyone()


def normalize_gradients(args, accelerator, network):
    if accelerator.sync_gradients and args.max_grad_norm != 0.0:
        params_to_clip = network.get_trainable_params()
        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)


def calculate_loss(args, accelerator, noise_scheduler, batch, timesteps, noise_pred, target):
    loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float(), reduction="none")
    loss = loss.mean([1, 2, 3])  # mean over H, W, C
    loss_weights = batch["loss_weights"]  # 各sampleごとのweight
    loss = loss * loss_weights

    if args.min_snr_gamma:
        loss = apply_snr_weight(loss, timesteps, noise_scheduler, args.min_snr_gamma)

    loss = loss.mean()  # 平均なのでbatch_sizeで割る必要なし
    loss = loss * args.v_noise_gamma
    accelerator.backward(loss)
    return loss


def encode_latents(args, tokenizer, accelerator, weight_dtype, text_encoder, vae, unet, train_text_encoder, on_step_start, batch):
    on_step_start(text_encoder, unet)

    with torch.no_grad():
        if "latents" in batch and batch["latents"] is not None:
            latents = batch["latents"].to(accelerator.device)
        else:
                        # latentに変換
            latents = vae.encode(batch["images"].to(dtype=weight_dtype)).latent_dist.sample()
        latents = latents * 0.18215
    b_size = latents.shape[0]

    with torch.set_grad_enabled(train_text_encoder):
                    # Get the text embedding for conditioning
        if args.weighted_captions:
            encoder_hidden_states = get_weighted_text_embeddings(
                            tokenizer,
                            text_encoder,
                            batch["captions"],
                            accelerator.device,
                            args.max_token_length // 75 if args.max_token_length else 1,
                            clip_skip=args.clip_skip,
                        )
        else:
            input_ids = batch["input_ids"].to(accelerator.device)
            encoder_hidden_states = train_util.get_hidden_states(args, input_ids, tokenizer, text_encoder, weight_dtype)
    return latents,b_size,encoder_hidden_states


def sample_and_predict_with_accumulate(args, tokenizer, accelerator, weight_dtype, text_encoder, vae, unet, network, train_text_encoder, optimizer, lr_scheduler, noise_scheduler, on_step_start, batch):
    """ Accumulate gradients over multiple steps. 
        """
    with accelerator.accumulate(network):
        latents, b_size, encoder_hidden_states = encode_latents(args, tokenizer, accelerator, weight_dtype, text_encoder, vae, unet, train_text_encoder, on_step_start, batch)

                # Sample noise that we'll add to the latents
        noise = sample_noise(args, latents)

                # Sample a random timestep for each image
        timesteps, noisy_latents = get_latents_from_noise(noise_scheduler, latents, b_size, noise)

                # Predict the noise residual
        noise_pred = noise_prediction(accelerator, unet, encoder_hidden_states, timesteps, noisy_latents)

        target = get_target_type(args, noise_scheduler, latents, noise, timesteps)

        loss = calculate_loss(args, accelerator, noise_scheduler, batch, timesteps, noise_pred, target)
        normalize_gradients(args, accelerator, network)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad(set_to_none=True)
    return loss


def perform_update(args, tokenizer, accelerator, unwrap_model, text_encoder, vae, unet, network, progress_bar, global_step, save_model, remove_model, epoch):
    progress_bar.update(1)
    global_step += 1

    train_util.sample_images(
                    accelerator, args, None, global_step, accelerator.device, vae, tokenizer, text_encoder, unet
                )

                # 指定ステップごとにモデルを保存
    if args.save_every_n_steps is not None and global_step % args.save_every_n_steps == 0:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            ckpt_name = train_util.get_step_ckpt_name(args, "." + args.save_model_as, global_step)
            save_model(ckpt_name, unwrap_model(network), global_step, epoch)

            if args.save_state:
                train_util.save_and_remove_state_stepwise(args, accelerator, global_step)

            remove_step_no = train_util.get_remove_step_no(args, global_step)
            if remove_step_no is not None:
                remove_ckpt_name = train_util.get_step_ckpt_name(args, "." + args.save_model_as, remove_step_no)
                remove_model(remove_ckpt_name)


def do_step(args, tokenizer, current_step, accelerator, unwrap_model, weight_dtype, text_encoder, vae, unet, network, train_text_encoder, optimizer, lr_scheduler, progress_bar, global_step, noise_scheduler, loss_list, loss_total, on_step_start, save_model, remove_model, epoch, step, batch):
    current_step.value = global_step
    loss = sample_and_predict_with_accumulate(args, tokenizer, accelerator, weight_dtype, text_encoder, vae, unet, network, train_text_encoder, optimizer, lr_scheduler, noise_scheduler, on_step_start, batch)

            # Checks if the accelerator has performed an optimization step behind the scenes
    if accelerator.sync_gradients:
        perform_update(args, tokenizer, accelerator, unwrap_model, text_encoder, vae, unet, network, progress_bar, global_step, save_model, remove_model, epoch)

    current_loss = loss.detach().item()
    if epoch == 0:
        loss_list.append(current_loss)
    else:
        loss_total -= loss_list[step]
        loss_list[step] = current_loss
    loss_total += current_loss
    avr_loss = loss_total / len(loss_list)
    logs = {"loss": avr_loss}  # , "lr": lr_scheduler.get_last_lr()[0]}
    progress_bar.set_postfix(**logs)

    do_logging(args, accelerator, lr_scheduler, global_step, current_loss, avr_loss)

class Trainer:
    
    def train(args):
        session_id = random.randint(0, 2**32)
        training_started_at = time.time()
        train_util.verify_training_args(args)
        train_util.prepare_dataset_args(args, True)

        cache_latents = args.cache_latents

        setup_rng(args)

        tokenizer = train_util.load_tokenizer(args)
        use_user_config, user_config = get_user_config(args)

        # データセットを準備する
        blueprint_generator = BlueprintGenerator(ConfigSanitizer(True, True, True))

        blueprint = blueprint_generator.generate(user_config, args, tokenizer=tokenizer)
        train_dataset_group = config_util.generate_dataset_group_by_blueprint(blueprint.dataset_group)

        current_epoch = Value("i", 0)
        current_step = Value("i", 0)
        ds_for_collater = train_dataset_group if args.max_data_loader_n_workers == 0 else None
        collater = train_util.collater_class(current_epoch, current_step, ds_for_collater)

        if args.debug_dataset:
            train_util.debug_dataset(train_dataset_group)
            return
        if len(train_dataset_group) == 0:
            print(
                "No data found. Please verify arguments (train_data_dir must be the parent of folders with images) / 画像がありません。引数指定を確認してください（train_data_dirには画像があるフォルダではなく、画像があるフォルダの親フォルダを指定する必要があります）"
            )
            return

        if cache_latents:
            assert (
                train_dataset_group.is_latent_cacheable()
            ), "when caching latents, either color_aug or random_crop cannot be used / latentをキャッシュするときはcolor_augとrandom_cropは使えません"

        # acceleratorを準備する
        print("prepare accelerator")
        accelerator, unwrap_model = train_util.prepare_accelerator(args)
        is_main_process = accelerator.is_main_process

        # mixed precisionに対応した型を用意しておき適宜castする
        weight_dtype, save_dtype = train_util.prepare_dtype(args)

        # モデルを読み込む
        text_encoder, vae, unet, _ = train_util.load_target_model(args, weight_dtype, accelerator)

        # モデルに xformers とか memory efficient attention を組み込む
        train_util.replace_unet_modules(unet, args.mem_eff_attn, args.xformers)

        # 学習を準備する
        if cache_latents:
            vae_latent_cache(args, train_dataset_group, accelerator, weight_dtype, vae)

        # prepare network

        net_kwargs, network = prepare_lora(args, text_encoder, vae, unet)

        train_unet = not args.network_train_text_encoder_only
        train_text_encoder = not args.network_train_unet_only
        network.apply_to(text_encoder, unet, train_text_encoder, train_unet)

        if args.network_weights is not None:
            info = network.load_weights(args.network_weights)
            print(f"load network weights from {args.network_weights}: {info}")

        if args.gradient_checkpointing:
            prepare_gradient_checkpointing(text_encoder, unet, network)  # may have no effect on network

        # 学習に必要なクラスを準備する
        print("prepare optimizer, data loader etc.")

        # 後方互換性を確保するよ
        try:
            trainable_params = network.prepare_optimizer_params(args.text_encoder_lr, args.unet_lr, args.learning_rate)
        except TypeError:
            print(
                "Deprecated: use prepare_optimizer_params(text_encoder_lr, unet_lr, learning_rate) instead of prepare_optimizer_params(text_encoder_lr, unet_lr)"
            )
            trainable_params = network.prepare_optimizer_params(args.text_encoder_lr, args.unet_lr)

        optimizer_name, optimizer_args, optimizer = train_util.get_optimizer(args, trainable_params)

        # dataloaderを準備する
        # DataLoaderのプロセス数：0はメインプロセスになる
        n_workers = min(args.max_data_loader_n_workers, os.cpu_count() - 1)  # cpu_count-1 ただし最大で指定された数まで

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset_group,
            batch_size=1,
            shuffle=True,
            collate_fn=collater,
            num_workers=n_workers,
            persistent_workers=args.persistent_data_loader_workers,
        )

        # 学習ステップ数を計算する
        if args.max_train_epochs is not None:
            args.max_train_steps = args.max_train_epochs * math.ceil(
                len(train_dataloader) / accelerator.num_processes / args.gradient_accumulation_steps
            )
            if is_main_process:
                print(f"override steps. steps for {args.max_train_epochs} epochs is / 指定エポックまでのステップ数: {args.max_train_steps}")

        # データセット側にも学習ステップを送信
        train_dataset_group.set_max_train_steps(args.max_train_steps)

        # lr schedulerを用意する
        lr_scheduler = train_util.get_scheduler_fix(args, optimizer, accelerator.num_processes)

        # 実験的機能：勾配も含めたfp16学習を行う　モデル全体をfp16にする
        if args.full_fp16:
            assert (
                args.mixed_precision == "fp16"
            ), "full_fp16 requires mixed precision='fp16' / full_fp16を使う場合はmixed_precision='fp16'を指定してください。"
            print("enable full fp16 training.")
            network.to(weight_dtype)

        # acceleratorがなんかよろしくやってくれるらしい
        text_encoder, unet, network, optimizer, train_dataloader, lr_scheduler = prepare_networks(accelerator, train_unet, train_text_encoder)

        # transform DDP after prepare (train_network here only)
        text_encoder, unet, network = train_util.transform_if_model_is_DDP(text_encoder, unet, network)

        unet.requires_grad_(False)
        unet.to(accelerator.device, dtype=weight_dtype)
        text_encoder.requires_grad_(False)
        text_encoder.to(accelerator.device)
        if args.gradient_checkpointing:  # according to TI example in Diffusers, train is required
            unet.train()
            text_encoder.train()

            # set top parameter requires_grad = True for gradient checkpointing works
            text_encoder.text_model.embeddings.requires_grad_(True)
        else:
            unet.eval()
            text_encoder.eval()

        network.prepare_grad_etc(text_encoder, unet)

        if not cache_latents:
            vae.requires_grad_(False)
            vae.eval()
            vae.to(accelerator.device, dtype=weight_dtype)

        # 実験的機能：勾配も含めたfp16学習を行う　PyTorchにパッチを当ててfp16でのgrad scaleを有効にする
        if args.full_fp16:
            train_util.patch_accelerator_for_fp16_training(accelerator)

        # resumeする
        train_util.resume_from_local_or_hf_if_specified(accelerator, args)

        # epoch数を計算する
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
        if (args.save_n_epoch_ratio is not None) and (args.save_n_epoch_ratio > 0):
            args.save_every_n_epochs = math.floor(num_train_epochs / args.save_n_epoch_ratio) or 1

        # 学習する
        # TODO: find a way to handle total batch size when there are multiple datasets
        total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

        if is_main_process:
            print_generation_information(args, train_dataset_group, train_dataloader, num_train_epochs)

        # TODO refactor metadata creation and move to util
        metadata = get_combined_metadata(args, session_id, training_started_at, use_user_config, train_dataset_group, net_kwargs, optimizer_name, optimizer_args, train_dataloader, num_train_epochs, total_batch_size)

        # make minimum metadata for filtering
        minimum_metadata = generate_metadata(metadata)

        progress_bar = tqdm(range(args.max_train_steps), smoothing=0, disable=not accelerator.is_local_main_process, desc="steps")
        global_step = 0

        noise_scheduler = DDPMScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, clip_sample=False, prediction_type="v_prediction"
        )

        patch_scheduler_betas(noise_scheduler)

        if accelerator.is_main_process:
            accelerator.init_trackers("network_train" if args.log_tracker_name is None else args.log_tracker_name)

        loss_list = []
        loss_total = 0.0
        del train_dataset_group

        # callback for step start
        if hasattr(network, "on_step_start"):
            on_step_start = network.on_step_start
        else:
            on_step_start = lambda *args, **kwargs: None

        # function for saving/removing
        def save_model(ckpt_name, unwrapped_nw, steps, epoch_no, force_sync_upload=False):
            os.makedirs(args.output_dir, exist_ok=True)
            ckpt_file = os.path.join(args.output_dir, ckpt_name)

            print(f"\nsaving checkpoint: {ckpt_file}")
            metadata["ss_training_finished_at"] = str(time.time())
            metadata["ss_steps"] = str(steps)
            metadata["ss_epoch"] = str(epoch_no)

            unwrapped_nw.save_weights(ckpt_file, save_dtype, minimum_metadata if args.no_metadata else metadata)
            if args.huggingface_repo_id is not None:
                huggingface_util.upload(args, ckpt_file, "/" + ckpt_name, force_sync_upload=force_sync_upload)

        def remove_model(old_ckpt_name):
            old_ckpt_file = os.path.join(args.output_dir, old_ckpt_name)
            if os.path.exists(old_ckpt_file):
                print(f"removing old checkpoint: {old_ckpt_file}")
                os.remove(old_ckpt_file)

        # training loop
        for epoch in range(num_train_epochs):
            if is_main_process:
                print(f"\nepoch {epoch+1}/{num_train_epochs}")
            current_epoch.value = epoch + 1

            metadata["ss_epoch"] = str(epoch + 1)

            network.on_epoch_start(text_encoder, unet)

            for step, batch in enumerate(train_dataloader):
                do_step(args, tokenizer, current_step, accelerator, unwrap_model, weight_dtype, text_encoder, vae, unet, network, train_text_encoder, optimizer, lr_scheduler, progress_bar, global_step, noise_scheduler, loss_list, loss_total, on_step_start, save_model, remove_model, epoch, step, batch)
                if global_step >= args.max_train_steps:
                    break

            if args.logging_dir is not None:
                logs = {"loss/epoch": loss_total / len(loss_list)}
                accelerator.log(logs, step=epoch + 1)

            accelerator.wait_for_everyone()

            # 指定エポックごとにモデルを保存
            if args.save_every_n_epochs is not None:
                save_files(args, accelerator, unwrap_model, is_main_process, network, num_train_epochs, global_step, save_model, remove_model, epoch)

            train_util.sample_images(accelerator, args, epoch + 1, global_step, accelerator.device, vae, tokenizer, text_encoder, unet)

            # end of epoch

        # metadata["ss_epoch"] = str(num_train_epochs)
        metadata["ss_training_finished_at"] = str(time.time())

        if is_main_process:
            network = unwrap_model(network)

        accelerator.end_training()

        if is_main_process and args.save_state:
            train_util.save_state_on_train_end(args, accelerator)

        del accelerator  # この後メモリを使うのでこれは消す

        if is_main_process:
            ckpt_name = train_util.get_last_ckpt_name(args, "." + args.save_model_as)
            save_model(ckpt_name, network, global_step, num_train_epochs, force_sync_upload=True)

            print("model saved.")
import library.train_util as train_util
from library.train_util import DreamBoothDataset


import json
import os


def generate_user_metadata(args, session_id, training_started_at, train_dataset_group, optimizer_name, optimizer_args, train_dataloader, num_train_epochs):
    metadata = {
        "ss_session_id": session_id,  # random integer indicating which group of epochs the model came from
        "ss_training_started_at": training_started_at,  # unix timestamp
        "ss_output_name": args.output_name,
        "ss_learning_rate": args.learning_rate,
        "ss_text_encoder_lr": args.text_encoder_lr,
        "ss_unet_lr": args.unet_lr,
        "ss_num_train_images": train_dataset_group.num_train_images,
        "ss_num_reg_images": train_dataset_group.num_reg_images,
        "ss_num_batches_per_epoch": len(train_dataloader),
        "ss_num_epochs": num_train_epochs,
        "ss_gradient_checkpointing": args.gradient_checkpointing,
        "ss_gradient_accumulation_steps": args.gradient_accumulation_steps,
        "ss_max_train_steps": args.max_train_steps,
        "ss_lr_warmup_steps": args.lr_warmup_steps,
        "ss_lr_scheduler": args.lr_scheduler,
        "ss_network_module": args.network_module,
        "ss_network_dim": args.network_dim,  # None means default because another network than LoRA may have another default dim
        "ss_network_alpha": args.network_alpha,  # some networks may not use this value
        "ss_mixed_precision": args.mixed_precision,
        "ss_full_fp16": bool(args.full_fp16),
        "ss_v2": bool(args.v2),
        "ss_clip_skip": args.clip_skip,
        "ss_max_token_length": args.max_token_length,
        "ss_cache_latents": bool(args.cache_latents),
        "ss_seed": args.seed,
        "ss_lowram": args.lowram,
        "ss_noise_offset": args.noise_offset,
        "ss_multires_noise_iterations": args.multires_noise_iterations,
        "ss_multires_noise_discount": args.multires_noise_discount,
        "ss_adaptive_noise_scale": args.adaptive_noise_scale,
        "ss_training_comment": args.training_comment,  # will not be updated after training
        "ss_sd_scripts_commit_hash": train_util.get_git_revision_hash(),
        "ss_optimizer": optimizer_name + (f"({optimizer_args})" if len(optimizer_args) > 0 else ""),
        "ss_max_grad_norm": args.max_grad_norm,
        "ss_caption_dropout_rate": args.caption_dropout_rate,
        "ss_caption_dropout_every_n_epochs": args.caption_dropout_every_n_epochs,
        "ss_caption_tag_dropout_rate": args.caption_tag_dropout_rate,
        "ss_face_crop_aug_range": args.face_crop_aug_range,
        "ss_prior_loss_weight": args.prior_loss_weight,
        "ss_min_snr_gamma": args.min_snr_gamma,
    }

    return metadata


def get_combined_metadata(args, session_id, training_started_at, use_user_config, train_dataset_group, net_kwargs, optimizer_name, optimizer_args, train_dataloader, num_train_epochs, total_batch_size):
    metadata = generate_user_metadata(args, session_id, training_started_at, train_dataset_group, optimizer_name, optimizer_args, train_dataloader, num_train_epochs)

    if use_user_config:
        # save metadata of multiple datasets
        # NOTE: pack "ss_datasets" value as json one time
        #   or should also pack nested collections as json?
        datasets_metadata = []
        tag_frequency = {}  # merge tag frequency for metadata editor
        dataset_dirs_info = {}  # merge subset dirs for metadata editor

        for dataset in train_dataset_group.datasets:
            is_dreambooth_dataset = isinstance(dataset, DreamBoothDataset)
            dataset_metadata = {
                "is_dreambooth": is_dreambooth_dataset,
                "batch_size_per_device": dataset.batch_size,
                "num_train_images": dataset.num_train_images,  # includes repeating
                "num_reg_images": dataset.num_reg_images,
                "resolution": (dataset.width, dataset.height),
                "enable_bucket": bool(dataset.enable_bucket),
                "min_bucket_reso": dataset.min_bucket_reso,
                "max_bucket_reso": dataset.max_bucket_reso,
                "tag_frequency": dataset.tag_frequency,
                "bucket_info": dataset.bucket_info,
            }

            subsets_metadata = []
            for subset in dataset.subsets:
                subset_metadata = {
                    "img_count": subset.img_count,
                    "num_repeats": subset.num_repeats,
                    "color_aug": bool(subset.color_aug),
                    "flip_aug": bool(subset.flip_aug),
                    "random_crop": bool(subset.random_crop),
                    "shuffle_caption": bool(subset.shuffle_caption),
                    "keep_tokens": subset.keep_tokens,
                }

                image_dir_or_metadata_file = None
                if subset.image_dir:
                    image_dir = os.path.basename(subset.image_dir)
                    subset_metadata["image_dir"] = image_dir
                    image_dir_or_metadata_file = image_dir

                if is_dreambooth_dataset:
                    subset_metadata["class_tokens"] = subset.class_tokens
                    subset_metadata["is_reg"] = subset.is_reg
                    if subset.is_reg:
                        image_dir_or_metadata_file = None  # not merging reg dataset
                else:
                    metadata_file = os.path.basename(subset.metadata_file)
                    subset_metadata["metadata_file"] = metadata_file
                    image_dir_or_metadata_file = metadata_file  # may overwrite

                subsets_metadata.append(subset_metadata)

                # merge dataset dir: not reg subset only
                # TODO update additional-network extension to show detailed dataset config from metadata
                if image_dir_or_metadata_file is not None:
                    # datasets may have a certain dir multiple times
                    v = image_dir_or_metadata_file
                    i = 2
                    while v in dataset_dirs_info:
                        v = image_dir_or_metadata_file + f" ({i})"
                        i += 1
                    image_dir_or_metadata_file = v

                    dataset_dirs_info[image_dir_or_metadata_file] = {"n_repeats": subset.num_repeats, "img_count": subset.img_count}

            dataset_metadata["subsets"] = subsets_metadata
            datasets_metadata.append(dataset_metadata)

            # merge tag frequency:
            for ds_dir_name, ds_freq_for_dir in dataset.tag_frequency.items():
                # あるディレクトリが複数のdatasetで使用されている場合、一度だけ数える
                # もともと繰り返し回数を指定しているので、キャプション内でのタグの出現回数と、それが学習で何度使われるかは一致しない
                # なので、ここで複数datasetの回数を合算してもあまり意味はない
                if ds_dir_name in tag_frequency:
                    continue
                tag_frequency[ds_dir_name] = ds_freq_for_dir

            
            
        metadata["ss_datasets"] = json.dumps(datasets_metadata)
        metadata["ss_tag_frequency"] = json.dumps(tag_frequency)
        metadata["ss_dataset_dirs"] = json.dumps(dataset_dirs_info)
    else:
        # conserving backward compatibility when using train_dataset_dir and reg_dataset_dir
        assert (
            len(train_dataset_group.datasets) == 1
        ), f"There should be a single dataset but {len(train_dataset_group.datasets)} found. This seems to be a bug. / データセットは1個だけ存在するはずですが、実際には{len(train_dataset_group.datasets)}個でした。プログラムのバグかもしれません。"

        dataset = train_dataset_group.datasets[0]

        dataset_dirs_info = {}
        reg_dataset_dirs_info = {}

        for subset in dataset.subsets:
            get_subset_info(args, dataset_dirs_info, subset, reg_dataset_dirs_info)
            
        metadata.update(
            {
                "ss_batch_size_per_device": args.train_batch_size,
                "ss_total_batch_size": total_batch_size,
                "ss_resolution": args.resolution,
                "ss_color_aug": bool(args.color_aug),
                "ss_flip_aug": bool(args.flip_aug),
                "ss_random_crop": bool(args.random_crop),
                "ss_shuffle_caption": bool(args.shuffle_caption),
                "ss_enable_bucket": bool(dataset.enable_bucket),
                "ss_bucket_no_upscale": bool(dataset.bucket_no_upscale),
                "ss_min_bucket_reso": dataset.min_bucket_reso,
                "ss_max_bucket_reso": dataset.max_bucket_reso,
                "ss_keep_tokens": args.keep_tokens,
                "ss_dataset_dirs": json.dumps(dataset_dirs_info),
                "ss_reg_dataset_dirs": json.dumps(reg_dataset_dirs_info),
            }
        )
        metadata.update(
            {
                "ss_batch_size_per_device": args.train_batch_size,
                "ss_total_batch_size": total_batch_size,
                "ss_resolution": args.resolution,
                "ss_color_aug": bool(args.color_aug),
                "ss_flip_aug": bool(args.flip_aug),
                "ss_random_crop": bool(args.random_crop),
                "ss_shuffle_caption": bool(args.shuffle_caption),
                "ss_enable_bucket": bool(dataset.enable_bucket),
                "ss_bucket_no_upscale": bool(dataset.bucket_no_upscale),
                "ss_min_bucket_reso": dataset.min_bucket_reso,
                "ss_max_bucket_reso": dataset.max_bucket_reso,
                "ss_keep_tokens": args.keep_tokens,
                "ss_dataset_dirs": json.dumps(dataset_dirs_info),
                "ss_reg_dataset_dirs": json.dumps(reg_dataset_dirs_info),
                "ss_tag_frequency": json.dumps(dataset.tag_frequency),
                "ss_bucket_info": json.dumps(dataset.bucket_info),
            }
        )

    # add extra args
    if args.network_args:
        metadata["ss_network_args"] = json.dumps(net_kwargs)

    # model name and hash
    if args.pretrained_model_name_or_path is not None:
        sd_model_name = args.pretrained_model_name_or_path
        if os.path.exists(sd_model_name):
            metadata["ss_sd_model_hash"] = train_util.model_hash(sd_model_name)
            metadata["ss_new_sd_model_hash"] = train_util.calculate_sha256(sd_model_name)
            sd_model_name = os.path.basename(sd_model_name)
        metadata["ss_sd_model_name"] = sd_model_name

    if args.vae is not None:
        vae_name = args.vae
        if os.path.exists(vae_name):
            metadata["ss_vae_hash"] = train_util.model_hash(vae_name)
            metadata["ss_new_vae_hash"] = train_util.calculate_sha256(vae_name)
            vae_name = os.path.basename(vae_name)
        metadata["ss_vae_name"] = vae_name

    metadata = {k: str(v) for k, v in metadata.items()}
    return metadata

def get_dataset_dirs_info(args, dataset_group):
    dataset_dirs_info = {}
    reg_dataset_dirs_info = {}
    for dataset in dataset_group.datasets:
        for subset in dataset.subsets:
            get_subset_info(args, dataset_dirs_info, subset, reg_dataset_dirs_info)
    return dataset_dirs_info, reg_dataset_dirs_info

# A complete redesign that is API comptaible with the original get_combined_metadata
def get_combined_metadata_redesign(args, dataset_group):
    metadata = {
            "ss_batch_size_per_device": args.train_batch_size,
            "ss_total_batch_size": total_batch_size,
            "ss_resolution": args.resolution,
            "ss_color_aug": bool(args.color_aug),
            "ss_flip_aug": bool(args.flip_aug),
            "ss_random_crop": bool(args.random_crop),
            "ss_shuffle_caption": bool(args.shuffle_caption),
            "ss_enable_bucket": bool(dataset.enable_bucket),
            "ss_bucket_no_upscale": bool(dataset.bucket_no_upscale),
            "ss_min_bucket_reso": dataset.min_bucket_reso,
            "ss_max_bucket_reso": dataset.max_bucket_reso,
            "ss_keep_tokens": args.keep_tokens,
        }

    # add extra args
    if args.network_args:
        metadata["ss_network_args"] = json.dumps(net_kwargs)

    # model name and hash
    if args.pretrained_model_name_or_path is not None:
        sd_model_name = args.pretrained_model_name_or_path
        if os.path.exists(sd_model_name):
            metadata["ss_sd_model_hash"] = train_util.model_hash(sd_model_name)
            metadata["ss_new_sd_model_hash"] = train_util.calculate_sha256(sd_model_name)
            sd_model_name = os.path.basename(sd_model_name)
        metadata["ss_sd_model_name"] = sd_model_name

    if args.vae is not None:
        vae_name = args.vae
        if os.path.exists(vae_name):
            metadata["ss_vae_hash"] = train_util.model_hash(vae_name)
            metadata["ss_new_vae_hash"] = train_util.calculate_sha256(vae_name)
            vae_name = os.path.basename(vae_name)
        metadata["ss_vae_name"] = vae_name

    # add datasets metadata
    datasets_metadata = {}
    tag_frequency = {}
    dataset_dirs_info = {}
    reg_dataset_dirs_info = {}
    for dataset in dataset_group.datasets:
        get_dataset_metadata(dataset, datasets_metadata, tag_frequency)
        for subset in dataset.subsets:
            get_subset_info(args, dataset_dirs_info, subset, reg_dataset_dirs_info)

    metadata["ss_datasets"] = json.dumps(datasets_metadata)
    metadata["ss_tag_frequency"] = json.dumps(tag_frequency)
    metadata["ss_dataset_dirs"] = json.dumps(dataset_dirs_info)
    metadata["ss_reg_dataset_dirs"] = json.dumps(reg_dataset_dirs_info)

    metadata = {k: str(v) for k, v in metadata.items()}
    return metadata

def get_dataset_metadata(dataset, datasets_metadata, tag_frequency):
    metadata = {
            "ss_dataset_name": dataset.name,
            "ss_dataset_type": dataset.type,
            "ss_dataset_dir": dataset.image_dir,
            "ss_dataset_metadata_file": dataset.metadata_file,
            "ss_dataset_n_repeats": dataset.num_repeats,
            "ss_dataset_img_count": dataset.img_count,
            "ss_dataset_tag_frequency": dataset.tag_frequency,
            "ss_dataset_bucket_info": dataset.bucket_info,
        }
    if dataset.type == "reg":
        metadata["ss_dataset_reg"] = True
    datasets_metadata[dataset.name] = metadata
    tag_frequency[dataset.name] = dataset.tag_frequency
    
def get_subset_info(args, dataset_dirs_info, subset, reg_dataset_dirs_info):
    if args.in_json:
        dataset_dirs_info[os.path.basename(subset.metadata_file)] = {
                    "n_repeats": subset.num_repeats,
                    "img_count": subset.img_count,
                }
    else:
        info = reg_dataset_dirs_info if subset.is_reg else dataset_dirs_info
        info[os.path.basename(subset.image_dir)] = {"n_repeats": subset.num_repeats, "img_count": subset.img_count}


def generate_metadata(metadata):
    minimum_keys = ["ss_network_module", "ss_network_dim", "ss_network_alpha", "ss_network_args"]
    minimum_metadata = {}
    for key in minimum_keys:
        if key in metadata:
            minimum_metadata[key] = metadata[key]
    return minimum_metadata

# A class based on the design of this file
class CombinedMetadata:
    def __init__(self, args, dataset_group, metadata=None):
        self.args = args
        self.dataset_group = dataset_group
        if metadata is None:
            self.metadata = self.get_combined_metadata()
        else:
            self.metadata = metadata

    def get_combined_metadata(self):
        metadata = {
                "ss_batch_size_per_device": self.args.train_batch_size,
                "ss_total_batch_size": total_batch_size,
                "ss_resolution": self.args.resolution,
                "ss_color_aug": bool(self.args.color_aug),
                "ss_flip_aug": bool(self.args.flip_aug),
                "ss_random_crop": bool(self.args.random_crop),
                "ss_shuffle_caption": bool(self.args.shuffle_caption),
                "ss_enable_bucket": bool(self.dataset_group.enable_bucket),
                "ss_bucket_no_upscale": bool(self.dataset_group.bucket_no_upscale),
                "ss_min_bucket_reso": self.dataset_group.min_bucket_reso,
                "ss_max_bucket_reso": self.dataset_group.max_bucket_reso,
                "ss_keep_tokens": self.args.keep_tokens,
            }

        # add extra args
        if self.args.network_args:
            metadata["ss_network_args"] = json.dumps(self.args.network_args)

        # model name and hash
        if self.args.pretrained_model_name_or_path is not None:
            sd_model_name = self.args.pretrained_model_name_or_path
            if os.path.exists(sd_model_name):
                metadata["ss_sd_model_hash"] = train_util.model_hash(sd_model_name)
                metadata["ss_new_sd_model_hash"] = train_util.calculate_sha256(sd_model_name)
                sd_model_name = os.path.basename(sd_model_name)
            metadata["ss_sd_model_name"] = sd_model_name

        if self.args.vae is not None:
            vae_name = self.args.vae
            if os.path.exists(vae_name):
                metadata["ss_vae_hash"] = train_util.model_hash(vae_name)
                metadata["ss_new_vae_hash"] = train_util.calculate_sha256(vae_name)
                vae_name = os.path.basename(vae_name)
            metadata["ss_vae_name"] = vae_name

        # add datasets metadata
        datasets_metadata = {}
        tag_frequency = {}
        dataset_dirs_info = {}
        reg_dataset_dirs_info = {}
        for dataset in self.dataset_group.datasets:
            self.get_dataset_metadata(dataset, datasets_metadata, tag_frequency)
            for subset in dataset.subsets:
                self.get_subset_info(dataset_dirs_info, subset, reg_dataset_dirs_info)
                
        metadata["ss_datasets"] = json.dumps(datasets_metadata)
        metadata["ss_tag_frequency"] = json.dumps(tag_frequency)
        metadata["ss_dataset_dirs"] = json.dumps(dataset_dirs_info)
        metadata["ss_reg_dataset_dirs"] = json.dumps(reg_dataset_dirs_info)

        metadata = {k: str(v) for k, v in metadata.items()}
        return metadata
    
    def get_dataset_metadata(self, dataset, datasets_metadata, tag_frequency):
        metadata = {
                "ss_dataset_name": dataset.name,
                "ss_dataset_type": dataset.type,
                "ss_dataset_dir": dataset.image_dir,
                "ss_dataset_metadata_file": dataset.metadata_file,
                "ss_dataset_n_repeats": dataset.num_repeats,
                "ss_dataset_img_count": dataset.img_count,
                "ss_dataset_tag_frequency": dataset.tag_frequency,
                "ss_dataset_bucket_info": dataset.bucket_info,
            }
        if dataset.type == "reg":
            metadata["ss_dataset_reg"] = True
        datasets_metadata[dataset.name] = metadata
        tag_frequency[dataset.name] = dataset.tag_frequency
        
    def get_subset_info(self, dataset_dirs_info, subset, reg_dataset_dirs_info):
        if self.args.in_json:
            dataset_dirs_info[os.path.basename(subset.metadata_file)] = {
                        "n_repeats": subset.num_repeats,
                        "img_count": subset.img_count,
                    }
        else:
            info = reg_dataset_dirs_info if subset.is_reg else dataset_dirs_info
            info[os.path.basename(subset.image_dir)] = {"n_repeats": subset.num_repeats, "img_count": subset.img_count}
    
    def generate_metadata(self):
        minimum_keys = ["ss_network_module", "ss_network_dim", "ss_network_alpha", "ss_network_args"]
        minimum_metadata = {}
        for key in minimum_keys:
            if key in self.metadata:
                minimum_metadata[key] = self.metadata[key]
        return minimum_metadata
    
    def save_metadata(self, metadata_file):
        metadata = self.generate_metadata()
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=4)
            
    def load_metadata(self, metadata_file):
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        self.metadata = metadata
        return metadata
    
    def get_metadata(self):
        return self.metadata
    
# The following are unimplemented portions of the class
    def get_dataset_metadata(self, dataset, datasets_metadata, tag_frequency):
        metadata = {
                "ss_dataset_name": dataset.name,
                "ss_dataset_type": dataset.type,
                "ss_dataset_dir": dataset.image_dir,
                "ss_dataset_metadata_file": dataset.metadata_file,
                "ss_dataset_n_repeats": dataset.num_repeats,
                "ss_dataset_img_count": dataset.img_count,
                "ss_dataset_tag_frequency": dataset.tag_frequency,
                "ss_dataset_bucket_info": dataset.bucket_info,
            }
        if dataset.type == "reg":
            metadata["ss_dataset_reg"] = True
        datasets_metadata[dataset.name] = metadata
        tag_frequency[dataset.name] = dataset.tag_frequency
        
    def get_subset_info(self, dataset_dirs_info, subset, reg_dataset_dirs_info):
        if self.args.in_json:
            dataset_dirs_info[os.path.basename(subset.metadata_file)] = {
                        "n_repeats": subset.num_repeats,
                        "img_count": subset.img_count,
                    }
        else:
            info = reg_dataset_dirs_info if subset.is_reg else dataset_dirs_info
            info[os.path.basename(subset.image_dir)] = {"n_repeats": subset.num_repeats, "img_count": subset.img_count}
            
    
        
import os
import torch
from copy import deepcopy
from omegaconf.dictconfig import DictConfig
from huggingface_hub import hf_hub_url, cached_download

from .configs import CONFIG_2_0, CONFIG_2_1
from .kandinsky2_model import Kandinsky2
from .kandinsky2_1_model import Kandinsky2_1
from .kandinsky2_2_model import Kandinsky2_2
from .kandinsky3_pipeline import Kandinsky3Pipeline

ckpt_dir = "weights"

def get_kandinsky2_0(device, task_type = "text2img", use_auth_token = None, use_flash_attention = False, ckpt_dir = ckpt_dir):
    cache_dir = os.path.join(ckpt_dir, "2_0")
    config = deepcopy(CONFIG_2_0)
    config["model_config"]["use_flash_attention"] = use_flash_attention
    if task_type == "inpainting":
        model_name = "Kandinsky-2-0-inpainting.pt"
        config_file_url = hf_hub_url(repo_id = "sberbank-ai/Kandinsky_2.0", filename = model_name)
    elif task_type == "text2img" or task_type == "img2img":
        model_name = "Kandinsky-2-0.pt"
        config_file_url = hf_hub_url(repo_id = "sberbank-ai/Kandinsky_2.0", filename = model_name)
    else:
        raise ValueError("                text2img, img2img   inpainting")
    cached_download(config_file_url, cache_dir = cache_dir, force_filename = model_name, use_auth_token = use_auth_token)
    cache_dir_text_en1 = os.path.join(cache_dir, "text_encoder1")
    for name in ["config.json", "pytorch_model.bin", "sentencepiece.bpe.model", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json"]:
        config_file_url = hf_hub_url(repo_id = "sberbank-ai/Kandinsky_2.0", filename = f"text_encoder1/{name}")
        cached_download(config_file_url, cache_dir = cache_dir_text_en1, force_filename = name, use_auth_token = use_auth_token)
    cache_dir_text_en2 = os.path.join(cache_dir, "text_encoder2")
    for name in ["config.json", "pytorch_model.bin", "spiece.model", "special_tokens_map.json", "tokenizer_config.json"]:
        config_file_url = hf_hub_url(repo_id = "sberbank-ai/Kandinsky_2.0", filename = f"text_encoder2/{name}")
        cached_download(config_file_url, cache_dir = cache_dir_text_en2, force_filename = name, use_auth_token = use_auth_token)
    config_file_url = hf_hub_url(repo_id = "sberbank-ai/Kandinsky_2.0", filename = "vae.ckpt")
    cached_download(config_file_url, cache_dir = cache_dir, force_filename = "vae.ckpt", use_auth_token = use_auth_token)
    config["text_enc_params1"]["model_path"] = cache_dir_text_en1
    config["text_enc_params2"]["model_path"] = cache_dir_text_en2
    config["tokenizer_name1"] = cache_dir_text_en1
    config["tokenizer_name2"] = cache_dir_text_en2
    config["image_enc_params"]["params"]["ckpt_path"] = os.path.join(cache_dir, "vae.ckpt")
    unet_path = os.path.join(cache_dir, model_name)
    model = Kandinsky2(config, unet_path, device, task_type)
    return model


def get_kandinsky2_1(device, task_type = "text2img", use_auth_token = None, use_flash_attention = False, ckpt_dir = ckpt_dir):
    cache_dir = os.path.join(ckpt_dir, "2_1")
    config = DictConfig(deepcopy(CONFIG_2_1))
    config["model_config"]["use_flash_attention"] = use_flash_attention
    if task_type == "text2img" or task_type == "img2img":
        model_name = "decoder_fp16.ckpt"
        config_file_url = hf_hub_url(repo_id = "sberbank-ai/Kandinsky_2.1", filename = model_name)
    elif task_type == "inpainting":
        model_name = "inpainting_fp16.ckpt"
        config_file_url = hf_hub_url(repo_id = "sberbank-ai/Kandinsky_2.1", filename = model_name)
    cached_download(config_file_url, cache_dir = cache_dir, force_filename = model_name, use_auth_token = use_auth_token)
    prior_name = "prior_fp16.ckpt"
    config_file_url = hf_hub_url(repo_id = "sberbank-ai/Kandinsky_2.1", filename = prior_name)
    cached_download(config_file_url, cache_dir = cache_dir, force_filename = prior_name, use_auth_token = use_auth_token)
    cache_dir_text_en = os.path.join(cache_dir, "text_encoder")
    for name in ["config.json", "pytorch_model.bin", "sentencepiece.bpe.model", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json"]:
        config_file_url = hf_hub_url(repo_id = "sberbank-ai/Kandinsky_2.1", filename = f"text_encoder/{name}")
        cached_download(config_file_url, cache_dir = cache_dir_text_en, force_filename = name, use_auth_token = use_auth_token)
    config_file_url = hf_hub_url(repo_id = "sberbank-ai/Kandinsky_2.1", filename = "movq_final.ckpt")
    cached_download(config_file_url, cache_dir=cache_dir, force_filename = "movq_final.ckpt", use_auth_token = use_auth_token)
    config_file_url = hf_hub_url(repo_id = "sberbank-ai/Kandinsky_2.1", filename = "ViT-L-14_stats.th")
    cached_download(config_file_url, cache_dir = cache_dir, force_filename = "ViT-L-14_stats.th", use_auth_token = use_auth_token)
    config["tokenizer_name"] = cache_dir_text_en
    config["text_enc_params"]["model_path"] = cache_dir_text_en
    config["prior"]["clip_mean_std_path"] = os.path.join(cache_dir, "ViT-L-14_stats.th")
    config["image_enc_params"]["ckpt_path"] = os.path.join(cache_dir, "movq_final.ckpt")
    cache_model_name = os.path.join(cache_dir, model_name)
    cache_prior_name = os.path.join(cache_dir, prior_name)
    model = Kandinsky2_1(config, cache_model_name, cache_prior_name, device, task_type = task_type)
    return model


def get_kandinsky(device, task_type = "text2img", use_auth_token = None, model_version = "2.2", use_flash_attention = False, torch_dtype = torch.float16, ckpt_dir = ckpt_dir):
    if model_version == "2.0":
        model = get_kandinsky2_0(device, task_type = task_type, use_auth_token = use_auth_token, use_flash_attention = use_flash_attention, ckpt_dir = ckpt_dir)
    elif model_version == "2.1":
        model = get_kandinsky2_1(device, task_type = task_type, use_auth_token = use_auth_token, use_flash_attention = use_flash_attention, ckpt_dir = ckpt_dir)
    elif model_version == "2.2":
        cache_dir = os.path.join(ckpt_dir, "2_2")
        model = Kandinsky2_2(device = device, task_type = task_type, cache_dir = cache_dir, torch_dtype = torch_dtype)
    elif model_version == "3.0":
        model = Kandinsky3Pipeline.from_pretrained(ckpt_dir + "\\3_0", variant = "fp16", torch_dtype = torch_dtype, cache_dir = ckpt_dir + "\\3_0", device_map = None, low_cpu_mem_usage = False)
    else:
        raise ValueError("                2.0, 2.1, 2.2   3.0")
    return model
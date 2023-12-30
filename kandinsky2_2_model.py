import io
import torch
from PIL import Image
import cv2
import numpy as np
from pytorch_lightning import seed_everything
from diffusers import KandinskyV22Pipeline, KandinskyV22PriorPipeline, KandinskyV22Img2ImgPipeline, KandinskyV22InpaintPipeline, KandinskyV22ControlnetPipeline, KandinskyV22PriorEmb2EmbPipeline, KandinskyV22ControlnetImg2ImgPipeline
from transformers import CLIPVisionModelWithProjection, pipeline
from diffusers.models import UNet2DConditionModel

class Kandinsky2_2:
    def __init__(self, device, task_type = "text2img", cache_dir = "weights\\2_2", torch_dtype = torch.float16):
        self.device = device
        self.task_type = task_type
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained("kandinsky-community/kandinsky-2-2-prior", subfolder = "image_encoder", cache_dir = cache_dir).to(torch_dtype).to(self.device)
        if task_type == "text2img":
            self.unet = UNet2DConditionModel.from_pretrained("kandinsky-community/kandinsky-2-2-decoder", subfolder = "unet", cache_dir = cache_dir).to(torch_dtype).to(self.device)
            self.prior = KandinskyV22PriorPipeline.from_pretrained("kandinsky-community/kandinsky-2-2-prior", image_encoder = self.image_encoder, torch_dtype = torch_dtype)
            self.prior = self.prior.to(self.device)
            self.decoder = KandinskyV22Pipeline.from_pretrained("kandinsky-community/kandinsky-2-2-decoder", unet = self.unet, torch_dtype = torch_dtype)
            self.decoder = self.decoder.to(self.device)
        elif task_type == "text2imgCN":
            self.unet = UNet2DConditionModel.from_pretrained("kandinsky-community/kandinsky-2-2-decoder", subfolder = "unet", cache_dir = cache_dir).to(torch_dtype).to(self.device)
            self.prior = KandinskyV22PriorPipeline.from_pretrained("kandinsky-community/kandinsky-2-2-prior", image_encoder = self.image_encoder, torch_dtype = torch_dtype)
            self.prior = self.prior.to(self.device)
            self.decoder = KandinskyV22ControlnetPipeline.from_pretrained("kandinsky-community/kandinsky-2-2-controlnet-depth", unet = self.unet, torch_dtype = torch_dtype, cache_dir = cache_dir)
            self.decoder = self.decoder.to(self.device)
        elif task_type == "inpainting":
            self.unet = UNet2DConditionModel.from_pretrained("kandinsky-community/kandinsky-2-2-decoder-inpaint", subfolder = "unet", cache_dir = cache_dir).to(torch_dtype).to(self.device)
            self.prior = KandinskyV22PriorPipeline.from_pretrained("kandinsky-community/kandinsky-2-2-prior", image_encoder = self.image_encoder, torch_dtype = torch_dtype, cache_dir = cache_dir)
            self.prior = self.prior.to(self.device)
            self.decoder = KandinskyV22InpaintPipeline.from_pretrained("kandinsky-community/kandinsky-2-2-decoder-inpaint", unet = self.unet, torch_dtype = torch_dtype, cache_dir = cache_dir)
            self.decoder = self.decoder.to(self.device)
        elif task_type == "img2img":
            self.unet = UNet2DConditionModel.from_pretrained("kandinsky-community/kandinsky-2-2-decoder", subfolder = "unet", cache_dir = cache_dir).to(torch_dtype).to(self.device)
            self.prior = KandinskyV22PriorPipeline.from_pretrained("kandinsky-community/kandinsky-2-2-prior", image_encoder = self.image_encoder, torch_dtype = torch_dtype, cache_dir = cache_dir)
            self.prior = self.prior.to(self.device)
            self.decoder = KandinskyV22Img2ImgPipeline.from_pretrained("kandinsky-community/kandinsky-2-2-decoder", unet = self.unet, torch_dtype = torch_dtype, cache_dir = cache_dir)
            self.decoder = self.decoder.to(self.device)
        elif task_type == "img2imgCN":
            self.prior = KandinskyV22PriorEmb2EmbPipeline.from_pretrained("kandinsky-community/kandinsky-2-2-prior", image_encoder = self.image_encoder, torch_dtype = torch_dtype, cache_dir = cache_dir)
            self.prior = self.prior.to(self.device)
            self.decoder = KandinskyV22ControlnetImg2ImgPipeline.from_pretrained("kandinsky-community/kandinsky-2-2-controlnet-depth", torch_dtype = torch_dtype, cache_dir = cache_dir)
            self.decoder = self.decoder.to(self.device)
        elif task_type == "depth2img":
            self.prior = KandinskyV22PriorPipeline.from_pretrained("kandinsky-community/kandinsky-2-2-prior", image_encoder = self.image_encoder, torch_dtype = torch_dtype, cache_dir = cache_dir)
            self.prior = self.prior.to(self.device)
            self.decoder = KandinskyV22ControlnetPipeline.from_pretrained("kandinsky-community/kandinsky-2-2-controlnet-depth", torch_dtype = torch_dtype, cache_dir = cache_dir)
            self.decoder = self.decoder.to(self.device)
        else:
            raise ValueError("Only text2img, img2img, inpainting, depth2img is available")
            
    def get_new_h_w(self, h, w):
        new_h = h // 64
        if h % 64 != 0:
            new_h += 1
        new_w = w // 64
        if w % 64 != 0:
            new_w += 1
        return new_h * 64, new_w * 64
    
    def generate_text2img(self, prompt, batch_size = 1, decoder_steps = 50, prior_steps = 25, decoder_guidance_scale = 4, prior_guidance_scale = 4, h = 512, w = 512, negative_prior_prompt = "", negative_decoder_prompt = "", seed = 42):
        seed_everything(seed)
        h, w = self.get_new_h_w(h, w)
        img_emb = self.prior(prompt = prompt, num_inference_steps = prior_steps, num_images_per_prompt = batch_size, guidance_scale = prior_guidance_scale, negative_prompt = negative_prior_prompt)
        negative_emb = self.prior(prompt = negative_decoder_prompt, num_inference_steps = prior_steps, num_images_per_prompt = batch_size, guidance_scale = prior_guidance_scale)
        if negative_decoder_prompt == "":
            negative_emb = negative_emb.negative_image_embeds
        else:
            negative_emb = negative_emb.image_embeds
        images_i = self.decoder(image_embeds = img_emb.image_embeds, negative_image_embeds = negative_emb, num_inference_steps = decoder_steps, height = h, width = w, guidance_scale = decoder_guidance_scale).images
        images = []
        for image in images_i:
            buf = io.BytesIO()
            img = image
            img.save(buf, format = "PNG")
            b_data = buf.getvalue()
            img.close
            images.append(b_data)
        return images

    def generate_text2imgCN(self, prompt, batch_size = 1, decoder_steps = 100, prior_steps = 25, decoder_guidance_scale = 4, prior_guidance_scale = 4, h = 1024, w = 1024, negative_prior_prompt = "", negative_decoder_prompt = "", seed = 42):
        seed_everything(seed)
        h, w = self.get_new_h_w(h, w)
        depth_estimator = pipeline("depth-estimation")
        image = depth_estimator(Image.new(mode = "RGB", size = (1024, 1024), color = (153, 153, 255)))["depth"]
        image = np.array(image)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis = 2)
        detected_map = torch.from_numpy(image).float() / 255.0
        hint = detected_map.permute(2, 0, 1).unsqueeze(0).half().to("cuda")
        img_emb, zero_image_emb = self.prior(prompt = prompt, num_inference_steps = prior_steps, num_images_per_prompt = batch_size, guidance_scale = prior_guidance_scale, negative_prompt = negative_prior_prompt).to_tuple()
        images_i = self.decoder(image_embeds = img_emb, negative_image_embeds = zero_image_emb, hint = hint, num_inference_steps = decoder_steps, height = h, width = w, guidance_scale = decoder_guidance_scale).images
        images = []
        for image in images_i:
            buf = io.BytesIO()
            img = image
            img.save(buf, format = "PNG")
            b_data = buf.getvalue()
            img.close
            images.append(b_data)
        return images

    def generate_img2img(self, prompt, binary_data, strength = 0.4, batch_size = 1, decoder_steps = 100, prior_steps = 25, decoder_guidance_scale = 4, prior_guidance_scale = 4, h = 1024, w = 1024, negative_prior_prompt = "", negative_decoder_prompt = "", seed = 42, custom_orig_size = False):
        seed_everything(seed)
        image = Image.open(io.BytesIO(binary_data)).convert("RGB")
        if custom_orig_size != True:
            w, h = image.size
        h, w = self.get_new_h_w(h, w)
        img_emb = self.prior(prompt = prompt, num_inference_steps = prior_steps, num_images_per_prompt = batch_size, guidance_scale = prior_guidance_scale, negative_prompt = negative_prior_prompt)
        negative_emb = self.prior(prompt = negative_prior_prompt, num_inference_steps = prior_steps, num_images_per_prompt = batch_size, guidance_scale = prior_guidance_scale)
        if negative_decoder_prompt == "":
            negative_emb = negative_emb.negative_image_embeds
        else:
            negative_emb = negative_emb.image_embeds
        images_i = self.decoder(image_embeds = img_emb.image_embeds, negative_image_embeds = negative_emb, num_inference_steps = decoder_steps, height = h, width = w, guidance_scale = decoder_guidance_scale, strength = strength, image = image).images
        images = []
        for image in images_i:
            buf = io.BytesIO()
            img = image
            img.save(buf, format = "PNG")
            b_data = buf.getvalue()
            img.close
            images.append(b_data)
        return images
    
    def generate_depth2img(self, prompt, binary_data, batch_size = 1, decoder_steps = 100, prior_steps = 25, decoder_guidance_scale = 4, prior_guidance_scale = 4, h = 1024, w = 1024, negative_prior_prompt = "", seed = 42, custom_orig_size = True):
        seed_everything(seed)
        h, w = self.get_new_h_w(h, w)
        depth_estimator = pipeline("depth-estimation")
        image = depth_estimator(Image.open(io.BytesIO(binary_data)).convert("RGB"))["depth"]
        image = np.array(image)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis = 2)
        detected_map = torch.from_numpy(image).float() / 255.0
        hint = detected_map.permute(2, 0, 1).unsqueeze(0).half().to("cuda")
        img_emb, zero_image_emb = self.prior(prompt = prompt, num_inference_steps = prior_steps, num_images_per_prompt = batch_size, guidance_scale = prior_guidance_scale, negative_prompt = negative_prior_prompt).to_tuple()
        images_i = self.decoder(image_embeds = img_emb, negative_image_embeds = zero_image_emb, hint = hint, num_inference_steps = decoder_steps, height = h, width = w, guidance_scale = decoder_guidance_scale).images
        images = []
        for image in images_i:
            buf = io.BytesIO()
            img = image
            img.save(buf, format = "PNG")
            b_data = buf.getvalue()
            img.close
            images.append(b_data)
        return images
    
    def generate_img2imgCN(self, prompt, binary_data, batch_size = 1, decoder_steps = 50, prior_steps = 25, decoder_guidance_scale = 4, prior_guidance_scale = 4, h = 768, w = 768, negative_prior_prompt = "", seed = 42, prior_strength = 0.85, negative_prior_strength = 1.0, strength = 0.5, custom_orig_size = True):
        seed_everything(seed)
        img = Image.open(io.BytesIO(binary_data))
        if custom_orig_size != True:
            w, h = img.size
        h, w = self.get_new_h_w(h, w)
        img = img.resize((w, h), resample = Image.Resampling.LANCZOS)
        depth_estimator = pipeline("depth-estimation")
        image = depth_estimator(img)["depth"]
        image = np.array(image)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis = 2)
        detected_map = torch.from_numpy(image).float() / 255.0
        hint = detected_map.permute(2, 0, 1).unsqueeze(0).half().to("cuda")
        img_emb = self.prior(prompt = prompt, num_images_per_prompt = batch_size, image = img, guidance_scale = prior_guidance_scale, strength = prior_strength)
        negative_emb = self.prior(prompt = negative_prior_prompt, num_images_per_prompt = batch_size, image = img, guidance_scale = prior_guidance_scale, strength = negative_prior_strength)
        images_i = self.decoder(image = img, strength = strength, image_embeds = img_emb.image_embeds, negative_image_embeds = negative_emb.image_embeds, hint = hint, num_inference_steps = decoder_steps, height = h, width = w, guidance_scale = decoder_guidance_scale).images
        images = []
        for image in images_i:
            buf = io.BytesIO()
            img = image
            img.save(buf, format = "PNG")
            b_data = buf.getvalue()
            img.close
            images.append(b_data)
        return images

    def mix_images(self, image1_binary_data, image2_binary_data, prompt_image1, prompt_image2, weights, batch_size = 1, decoder_steps = 50, prior_steps = 25, decoder_guidance_scale = 4, prior_guidance_scale = 4, h = 1024, w = 1024, custom_orig_size = True, negative_prior_prompt = "", negative_decoder_prompt = "", seed = 42):
        seed_everything(seed)
        img1 = Image.open(io.BytesIO(image1_binary_data)).convert("RGB")
        img2 = Image.open(io.BytesIO(image2_binary_data)).convert("RGB")
        if custom_orig_size != True:
            w1, h1 = img1.size
            w2, h2 = img2.size
            w = max(w1, w2)
            h = max(h1, h2)
        new_h = h // 64     
        if h % 64 != 0:
            new_h += 1
        new_w = w // 64
        if w % 64 != 0:
            new_w += 1
        h = new_h * 64
        w = new_w * 64
        img1 = img1.resize((w, h), resample = Image.Resampling.LANCZOS)
        img2 = img2.resize((w, h), resample = Image.Resampling.LANCZOS)
        images_texts = [prompt_image1, img1, img2, prompt_image2]
        assert len(images_texts) == len(weights) and len(images_texts) > 0
        img_emb = self.prior.interpolate(images_and_prompts = images_texts, weights = weights, num_inference_steps = prior_steps, num_images_per_prompt = batch_size, guidance_scale = prior_guidance_scale, negative_prompt = negative_prior_prompt)
        negative_emb = self.prior(prompt = negative_prior_prompt, num_inference_steps = prior_steps, num_images_per_prompt = batch_size, guidance_scale = prior_guidance_scale)
        if negative_decoder_prompt == "":
            negative_emb = negative_emb.negative_image_embeds
        else:
            negative_emb = negative_emb.image_embeds
        images_i = self.decoder(image_embeds = img_emb.image_embeds, negative_image_embeds = negative_emb, num_inference_steps = decoder_steps, height = h, width = w, guidance_scale = decoder_guidance_scale).images
        images = []
        for image in images_i:
            buf = io.BytesIO()
            img = image
            img.save(buf, format = "PNG")
            b_data = buf.getvalue()
            img.close
            images.append(b_data)
        return images

    def generate_inpainting(self, prompt, binary_data, mask_binary_data, batch_size = 1, decoder_steps = 50, prior_steps = 25, decoder_guidance_scale = 4, prior_guidance_scale = 4, negative_prior_prompt = "", negative_decoder_prompt = "", seed = 42):
        seed_everything(seed)
        pil_img = Image.open(io.BytesIO(binary_data)).convert("RGB")
        img_mask = Image.open(io.BytesIO(mask_binary_data)).convert("RGBA")
        w, h = pil_img.size
        new_h = h // 64     
        if h % 64 != 0:
            new_h += 1
        new_w = w // 64
        if w % 64 != 0:
            new_w += 1
        h = new_h * 64
        w = new_w * 64
        pil_img = pil_img.resize((w, h), resample = Image.Resampling.LANCZOS)
        img_mask = img_mask.resize((w, h), resample = Image.Resampling.LANCZOS)
        img_array = np.array(img_mask)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2RGBA)
        array_mask = np.zeros((h, w), dtype = np.float32)
        array_mask = img_array[:, :, 3] / 255
        img_emb = self.prior(prompt = prompt, num_inference_steps = prior_steps, num_images_per_prompt = batch_size, guidance_scale = prior_guidance_scale, negative_prompt = negative_prior_prompt)
        negative_emb = self.prior(prompt = negative_prior_prompt, num_inference_steps = prior_steps, num_images_per_prompt = batch_size, guidance_scale = prior_guidance_scale)
        if negative_decoder_prompt == "":
            negative_emb = negative_emb.negative_image_embeds
        else:
            negative_emb = negative_emb.image_embeds
        images_i = self.decoder(image_embeds = img_emb.image_embeds, negative_image_embeds = negative_emb, num_inference_steps = decoder_steps, height = h, width = w, guidance_scale = decoder_guidance_scale, image = pil_img, mask_image = array_mask).images
        images = []
        for image in images_i:
            buf = io.BytesIO()
            img = image
            img.save(buf, format = "PNG")
            b_data = buf.getvalue()
            img.close
            images.append(b_data)
        return images
    
    def generate_stylization(self, prompt, content_binary_data, style_binary_data, batch_size = 1, decoder_steps = 50, prior_steps = 25, decoder_guidance_scale = 4, prior_guidance_scale = 4, negative_prior_prompt = "", style_size_as_content = True, prompt_weight = 0.3, content_weight = 0.3, style_weight = 0.4, seed = 42):
        seed_everything(seed)
        img1 = Image.open(io.BytesIO(content_binary_data)).convert("RGB")
        img2 = Image.open(io.BytesIO(style_binary_data)).convert("RGB")
        w, h = img1.size
        new_h = h // 64     
        if h % 64 != 0:
            new_h += 1
        new_w = w // 64
        if w % 64 != 0:
            new_w += 1
        h = new_h * 64
        w = new_w * 64
        img1 = img1.resize((w, h), resample = Image.Resampling.LANCZOS)
        if style_size_as_content == True:
            img2 = img2.resize((w, h), resample = Image.Resampling.LANCZOS)
        images_texts = [prompt, img1, img2]
        weights = [prompt_weight, content_weight, style_weight]
        prior_out = self.prior.interpolate(images_texts, weights, num_inference_steps = prior_steps, num_images_per_prompt = batch_size, guidance_scale = prior_guidance_scale, negative_prompt = negative_prior_prompt)
        images_i = self.decoder(**prior_out, num_inference_steps = decoder_steps, height = h, width = w, guidance_scale = decoder_guidance_scale).images
        images = []
        for image in images_i:
            buf = io.BytesIO()
            img = image
            img.save(buf, format = "PNG")
            b_data = buf.getvalue()
            img.close
            images.append(b_data)
        return images
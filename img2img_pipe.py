import torch
from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionImg2ImgPipeline


def img2img(init_image,prompt,strength):
    device = "cuda"
    model_id_or_path = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
    pipe = pipe.to(device)
    pipe.safety_checker = lambda images, clip_input: (images, False)

    images = pipe(prompt=prompt, image=init_image, strength=strength, guidance_scale=7.5).images

    return images[0]

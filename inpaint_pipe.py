from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import torch
import torch
import requests
from clipseg.models.clipseg import CLIPDensePredT
import cv2
from PIL import Image
from torchvision import transforms

def inpaint(image,mask_image,prompt,guidance_scale):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        revision="fp16",
        torch_dtype=torch.float16,
    ).to(device)
    pipe.safety_checker = lambda images, clip_input: (images, False)
    #image and mask_image should be PIL images.
    #The mask structure is white for inpainting and black for keeping as is
    image = pipe(prompt=prompt, image=image, mask_image=mask_image,guidance_scale=guidance_scale).images[0]
    return image

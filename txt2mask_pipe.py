from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import torch
import cv2
import numpy as np
from clipseg.models.clipseg import CLIPDensePredT
from PIL import Image,ImageChops
from torchvision import transforms
from matplotlib import pyplot as plt


def merge_masks(positive, negative):
    bw_image = cv2.cvtColor(
        np.array(ImageChops.darker(Image.fromarray(positive), Image.fromarray(negative))),
        cv2.COLOR_BGR2RGB)
    return cv2.cvtColor(bw_image, cv2.COLOR_BGR2GRAY)


def txt2mask(image, prompts, mode="Positive"):
    model = CLIPDensePredT(version='ViT-B/16', reduce_dim=64)
    model.eval()
    model.load_state_dict(torch.load('clipseg/weights/rd64-uni.pth', map_location=torch.device('cuda')), strict=False)

    # image and mask_image should be PIL images.
    # The mask structure is white for inpainting and black for keeping as is

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize((512, 512)),
    ])
    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        preds = model(img.repeat(len(prompts), 1, 1, 1), prompts)[0]

    _, ax = plt.subplots(1, 5, figsize=(15, 4))
    [a.axis('off') for a in ax.flatten()]
    ax[0].imshow(image)
    [ax[i + 1].imshow(torch.sigmoid(preds[i][0])) for i in range(len(prompts))];
    [ax[i + 1].text(0, -15, prompts[i]) for i in range(len(prompts))];

    filename = f"result_mask.png"
    plt.imsave(filename, torch.sigmoid(preds[0][0]))

    img2 = cv2.imread(filename)

    gray_image = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    (thresh, bw_image) = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY)

    if (mode == "Negative"):
        bw_image = np.invert(bw_image)

    cv2.cvtColor(bw_image, cv2.COLOR_BGR2RGB)

    Image.fromarray(bw_image)
    # For debugging only:
    cv2.imwrite(filename, bw_image)

    return bw_image


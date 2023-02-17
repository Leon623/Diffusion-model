import cv2
import argparse, os, sys, glob
from inpaint_pipe import inpaint
from PIL import Image

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        type=str,
        nargs="?",
        help="Image you want to inpaint"
    )

    parser.add_argument(
        "--n_outputs",
        type=int,
        nargs="?",
        default=10,
        help="How many images you want"
    )

    parser.add_argument(

        "--mask_file",
        type=str,
        nargs="?",
        help="Path to mask image"
    )

    parser.add_argument(

        "--output_dir",
        type=str,
        nargs="?",
        default="outputs",
        help="Name of output directory"
    )

    parser.add_argument(

        "--guidance_scale",
        type=float,
        nargs="?",
        default=7.5,
        help="Guidance for inpaint strength"
    )

    parser.add_argument(

        "--prompt",
        type=str,
        nargs="?",
        help="Prompt for inpaint"
    )

    opt = parser.parse_args()

    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)

    for _ in range(opt.n_outputs):
        input_image = Image.open(opt.image)

        mask_image = Image.open(opt.mask_file)

        result_image = inpaint(image=input_image, mask_image=mask_image, prompt=opt.prompt,
                               guidance_scale=opt.guidance_scale)

        print(f"Image {_} donr.")
        image_path = opt.output_dir
        result_image.save(f"{image_path}/" + f"{_}"+".jpg")

main()

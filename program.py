import argparse, os, sys, glob
import random
import string
import cv2
from PIL import Image
from inpaint_pipe import inpaint
from txt2mask_pipe import txt2mask
from img2img_pipe import img2img


def randStr(chars=string.ascii_uppercase + string.digits, N=10):
    return ''.join(random.choice(chars) for _ in range(N))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        type=str,
        nargs="?",
        help="Image you want to inpaint"
    )

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        help="Prompt you want to inpaint"
    )

    parser.add_argument(

        "--strength",
        type=float,
        nargs="?",
        default=0.6,
        help="Strength of transitioning"
    )

    parser.add_argument(

        "--out_file",
        type=str,
        nargs="?",
        default=randStr() + ".jpg",
        help="Name of output image"
    )

    parser.add_argument(

        "--output_dir",
        type=str,
        nargs="?",
        default="outputs",
        help="Name of output directory"
    )

    parser.add_argument(

        "--mask_output_dir",
        type=str,
        nargs="?",
        default="mask_outputs",
        help="Name of mask output directory"
    )

    parser.add_argument(

        "--transition_output_dir",
        type=str,
        nargs="?",
        default="transition_outputs",
        help="Name of transitions output directory"
    )

    parser.add_argument(

        "--guidance_scale",
        type=float,
        nargs="?",
        default=7.5,
        help="Higher guidance strength results in images closer to text prompt"
    )

    opt = parser.parse_args()
    print("Making transition")
    transition_image = img2img(init_image=Image.open(opt.image), prompt=opt.prompt, strength=opt.strength)
    print("Saving transition")
    image_path = opt.transition_output_dir
    transition_image.save(f"{image_path}/" + opt.out_file)
    print("Making mask")
    mask = txt2mask(transition_image, [opt.prompt])
    image_path = opt.mask_output_dir
    print("Saving mask")
    filename = f"{image_path}/" + opt.out_file
    cv2.imwrite(filename, mask)
    print("Making inpaint")
    image = inpaint(Image.open(opt.image), mask, opt.prompt, opt.guidance_scale)
    print("Saving inpaint")
    image_path = opt.output_dir
    image.save(f"{image_path}/" + opt.out_file)
    print(opt.out_file)
    print("Done.")


main()

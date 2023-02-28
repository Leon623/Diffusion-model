import cv2
import argparse, os, sys, glob
from inpaint_pipe import inpaint
from txt2mask_pipe import txt2mask
from img2img_pipe import img2img
from preprocess import preprocess_image, preprocess_data
from PIL import Image
from filter import filter_images, is_in_filter


def make_directory(directory_name):
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_dir",
        type=str,
        nargs="?",
        help="Directory of images you want to inpaint"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        nargs="?",
        help="Output directory of images to inpaint"
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
        default="Sunglasses",
        help="Prompt for inpaint"
    )

    parser.add_argument(

        "--strength",
        type=float,
        nargs="?",
        default=0.65,
        help="Strength of transitioning"
    )

    parser.add_argument(

        "--n_images",
        type=int,
        nargs="?",
        default=1,
        help="How many of each image"
    )

    parser.add_argument(

        "--images_filter",
        type=str,
        nargs="?",
        default="",
        help="Path to file of filtered images"
    )

    parser.add_argument(

        "--mask",
        action="store_true",
        help="If image already contains a mask"
    )

    parser.add_argument(

        "--starting_image",
        type=int,
        nargs="?",
        default=0,
        help="Starting number of inpaints"
    )

    opt = parser.parse_args()
    print("Starting...")
    transition_dir = opt.output_dir + "/transition_outputs"
    mask_dir = opt.output_dir + "/mask_outputs"
    inpaint_dir = opt.output_dir + "/inpaint_outputs"
    preprocess_dir = opt.output_dir + "/preprocessed_images"
    results_dir = opt.output_dir + "/result_images"
    make_directory(opt.output_dir)
    make_directory(transition_dir)
    make_directory(mask_dir)
    make_directory(inpaint_dir)
    make_directory(preprocess_dir)
    make_directory(results_dir)

    # Preprocessing images
    if not opt.mask:
        preprocess_data(source_directory=opt.input_dir, results_directory=preprocess_dir, resize_size=512)

    if opt.images_filter:
        filtered_images = filter_images(opt.images_filter)
    for image in os.listdir(preprocess_dir):
        if opt.images_filter:
            if not is_in_filter(image, filtered_images):
                print(f"{image} is not in filtered images")
                continue

        print(f"Processing image {image}")

        # Opening image
        input_image = Image.open(f"{preprocess_dir}/" + image)

        if not opt.mask:
            # Making transition
            transition_image = img2img(init_image=input_image, prompt=opt.prompt, strength=opt.strength)
            transition_image.save(f"{transition_dir}/" + image)

            # Masking image
            mask = txt2mask(transition_image, [opt.prompt])
            cv2.imwrite(f"{mask_dir}/" + image, mask)

        else:
            mask = Image.open(f"{mask_dir}/{image}")

        # Inpainting image
        for _ in range(opt.starting_image, opt.starting_image + opt.n_images):
            result_image = inpaint(input_image, mask, opt.prompt, opt.guidance_scale)
            result_image.save(f"{inpaint_dir}/{_}_" + image)

            print(f"Image {image} done.")

    # Convert inpaints back to results
    preprocess_data(source_directory=f"{inpaint_dir}/", results_directory=results_dir, resize_size=112)

main()


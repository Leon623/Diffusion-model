import cv2
import argparse, os, sys, glob
from inpaint_pipe import inpaint
from txt2mask_pipe import txt2mask
from img2img_pipe import img2img
from preprocess import preprocess_image, preprocess_data
from PIL import Image
print("asdasd")

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
        default=0.6,
        help="Strength of transitioning"
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

    #Preprocessing images
    preprocess_data(source_directory=opt.input_dir, results_directory=preprocess_dir, resize_size=512)
    for image in os.listdir(preprocess_dir):
        print(f"Processing image {image}")

        #Opening image
        input_image = Image.open(f"{preprocess_dir}/" + image)

        #Making transition
        transition_image = img2img(init_image=input_image, prompt=opt.prompt, strength=opt.strength)
        transition_image.save(f"{transition_dir}/" + image)

        #Masking image
        mask = txt2mask(transition_image, [opt.prompt])
        cv2.imwrite(f"{mask_dir}/" + image, mask)

        #Inpainting image
        result_image = inpaint(input_image, mask, opt.prompt, opt.guidance_scale)
        result_image.save(f"{inpaint_dir}/" + image)

        print(f"Image {image} done.")

    #Convert inpaints back to results
    preprocess_data(source_directory=f"{inpaint_dir}/", results_directory=results_dir, resize_size=250)


main()

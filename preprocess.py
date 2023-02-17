import torch
import torchvision.transforms.functional as fn
import os
from PIL import Image


# Funkcija za obradu slika
def preprocess_image(img, resize_size):
    if (img.width > img.height):
        img = fn.center_crop(img, output_size=[img.height, img.width])
    else:
        img = fn.center_crop(img, output_size=[img.width, img.width])

    if (resize_size):
        img = fn.resize(img, size=[resize_size, resize_size])

    return img


def preprocess_data(source_directory, results_directory, resize_size):
    for image in os.listdir(source_directory):
        img = Image.open(source_directory + str(image))
        img = preprocess_image(img, resize_size)
        img.save(results_directory + "/" + image)


# if __name__ == "__main__":
#     preprocess_data(source_directory="Paul_outputs/inpaint_outputs/", results_directory="Paul_outputs/result_outputs",
#                     resize_size=250)

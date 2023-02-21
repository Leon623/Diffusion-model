import os

def filter_images(input_file):
    file = open(input_file)
    images = [image.strip().split(":")[0].split("/")[1].split("_") for image in file.readlines()]

    dictionary = {image[0]: [] for image in images}
    for image in images:
        dictionary[image[0]].append(image[1])

    return dictionary


def is_in_filter(image, filtered_images):
    image_parts = image.split("_")

    if (image_parts[0] not in filtered_images):
        return False
    if (image_parts[1] not in filtered_images[image_parts[0]]):
        return False

    return True


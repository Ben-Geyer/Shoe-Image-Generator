import os
import numpy as np
from PIL import Image

IMAGE_SIZE = 128
IMAGE_DIR = "images/full"

def main():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), IMAGE_DIR)
    training_set = []

    for file in os.listdir(path):
        img_path = os.path.join(path, file)

        if os.path.isfile(img_path):
            image = Image.open(img_path).resize((IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)
            training_set.append(np.asarray(image))

    training_set = np.reshape(training_set, (-1, IMAGE_SIZE, IMAGE_SIZE, 3))
    training_set = (training_set / 127.5) - 1
    np.save("training_images.npy", training_set)

if __name__ == "__main__":
    main()

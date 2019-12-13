import math
import numpy as np
import matplotlib.pyplot as plt


def main():
    images = np.load("training_images.npy").astype("float32")[:16]
    side_size = math.sqrt(images[0].shape[0])
    fig = plt.figure(figsize = (side_size, side_size))

    for i in range(images.shape[0]):
        plt.subplot(side_size, side_size, i + 1)
        plt.imshow(((images[i] + 1) * 127.5).astype(np.int32))
        plt.axis("off")

    plt.show()

if __name__ == "__main__":
    main()

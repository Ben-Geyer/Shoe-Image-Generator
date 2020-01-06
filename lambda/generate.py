from tensorflow.python.keras.models import load_model
from PIL import Image
from hashlib import sha1
import numpy as np
import boto3
import json
import os


MODEL_S3_BUCKET = ""
IMAGE_S3_BUCKET = ""

# Clean /tmp of model
if os.path.exists("/tmp/model.h5"):
    os.remove("/tmp/model.h5")

# Download and load model
s3 = boto3.client("s3")
s3.download_file(MODEL_S3_BUCKET, "model.h5", "/tmp/model.h5")
model = load_model("/tmp/model.h5")
os.remove("/tmp/model.h5")

def gen_noise(num_samples):
    return np.random.normal(0, 1, size = (num_samples, 100))

def generate_handler(event, context):
    num_images = 1
    gen_images = model.predict(gen_noise(num_images))

    # Convert image from float representation to int
    gen_images = (gen_images * 127.5) + 127.5
    gen_images = gen_images.astype(np.uint8)

    gen_list = []

    for img in gen_images:
        # Generate unique name
        img_name = sha1(img).hexdigest() + ".jpg"

        # Save and upload image
        Image.fromarray(img, mode = "RGB").save("/tmp/" + img_name)
        s3.upload_file("/tmp/" + img_name, IMAGE_S3_BUCKET, "images/" + img_name)

        gen_list.append({
            "name": img_name,
            "path": "/" + IMAGE_S3_BUCKET + "/images/" + img_name
        })

    # Clean /tmp of model
    for fname in os.listdir("/tmp"):
        if fname.endswith(".jpg"):
            os.remove("/tmp/" + fname)

    return {
        "statusCode": 200,
        "body": json.dumps({
            "message": "Generated {} image(s)".format(num_images),
            "images": gen_list
        })
    }

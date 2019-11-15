import base64
import io
import os
import warnings

import h5py
import numpy as np
from PIL import Image

import utils


def load_assets(nsfw_model_path):
    global nsfw_model_bytes

    with open(nsfw_model_path, "rb") as nsfw_model_file:
        nsfw_model_bytes = nsfw_model_file.read()


def init():
    global keras, nsfw_model

    warnings.simplefilter("ignore")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")

    import numpy as np
    import h5py
    import tensorflow.keras as keras

    nsfw_model = keras.models.load_model(
        h5py.File(io.BytesIO(nsfw_model_bytes)),
    )

    # Warm up to make sure the model is responsive on first use. It seems like
    # Keras / Tensorflow does some lazy initialization on first predict call.
    shape = tuple([1, *nsfw_model.input_shape[1:]])
    nsfw_model.predict(np.zeros(shape, dtype=np.float32))


def wait_for_init():
    pass


def is_toxic_text(text):
    # TODO: Plug in Anthony's model.
    return False


def is_toxic_image(image):
    image = load_image(image)

    pred = nsfw_model.predict(image)[0]
    print("pred", pred)

    return pred.argmax() not in (0, 2)


def load_image(image):
    shape = tuple(nsfw_model.input_shape[1:3])

    image = utils.pipeline(
        base64.b64decode,
        io.BytesIO,
        Image.open,
        lambda x: x.resize(shape, Image.ANTIALIAS),
        keras.preprocessing.image.img_to_array,
    )(image)

    return np.array([image[:, :, :3] / 255])

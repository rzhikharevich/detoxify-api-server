import base64
import io
import os
import warnings

import h5py
import numpy as np
from PIL import Image
import joblib

import utils

TEXT_LABELS = ("toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate")
TEXT_MODEL_THRESHOLD = 0.7


def load_assets(nsfw_model_path, text_model_path):
    global nsfw_model_bytes, text_model_bytes, text_model_tk

    with open(nsfw_model_path, "rb") as nsfw_model_file:
        nsfw_model_bytes = nsfw_model_file.read()

    with open(f"{text_model_path}/network.hdf5", "rb") as text_model_file:
        text_model_bytes = text_model_file.read()

    text_model_tk = joblib.load(f"{text_model_path}/tokenizer.joblib")


def init():
    global keras, nsfw_model, text_model, text_model_tk

    warnings.simplefilter("ignore")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")

    import numpy as np
    import h5py
    import tensorflow.keras as keras

    nsfw_model = model_from_bytes(nsfw_model_bytes)

    text_model = model_from_bytes(text_model_bytes)

    # Warm up to make sure the model is responsive on first use. It seems like
    # Keras / Tensorflow does some lazy initialization on first predict call.

    shape = tuple([1, *nsfw_model.input_shape[1:]])
    nsfw_model.predict(np.zeros(shape, dtype=np.float32))

    _ = is_toxic_text("hello world")


def model_from_bytes(b):
    return keras.models.load_model(h5py.File(io.BytesIO(b)))


def wait_for_init():
    pass


def is_toxic_text(text):
    pred = utils.pipeline(
        text_model_tk.texts_to_sequences,
        lambda x: keras.preprocessing.sequence.pad_sequences(x, maxlen=150),
        text_model.predict
    )([text])

    return pred.max() > TEXT_MODEL_THRESHOLD


def is_toxic_image(image):
    image = load_image(image)

    pred = nsfw_model.predict(image)[0]

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


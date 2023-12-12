import numpy as np
import keras as tfk
from src.const import SIZE, SIZE_FULL
from src.crop import crop3x3



def concat(imgs: list[np.ndarray])->np.ndarray:
    return np.concatenate(
        [np.concatenate([
                imgs[i*3+j].reshape((1, SIZE, SIZE, 1)) for j in range(3)
                ],
            axis=1,
            ) for i in range(3)
        ], axis=2)[0]

def predict(model: tfk.Model, image: np.ndarray)->np.ndarray:
    """Predicts on 768x768x3 images (arrays of [0-1])

    Args:
        model (tfk.Model): Keras model for use
        image (np.ndarray): Input image of 768x768x3 shape

    Returns:
        np.ndarray: Out mask of 768x768x3 shape, 0. or 1. values
    """
    images = np.array([crop3x3(image, i) for i in range(9)])
    masks = model.predict_on_batch(images)
    return concat(masks).round()
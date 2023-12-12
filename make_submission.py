import numpy as np 
import keras as tfk
import os
import gc
from tqdm import tqdm

from src.crop import crop3x3
from src.const import *
from model import UNET_model
from src.encode_decode import encode


MODEL_FILE = "models\model-1-BCE+DiceLoss_val_score=0.64_val_loss=0.41.h5" # TODO: change model
SUBMISSON_FILE = f"submissions/submission_{MODEL_FILE}.csv"
TEST_PATH = "/kaggle/input/airbus-ship-detection/test_v2" 


print("Make sure you changed model file, and dataset path in code.")
print(f"Used model for submission file generation: {MODEL_FILE}")
print(f"Dataset path for generating submission: {TEST_PATH}")


def process(imgs: list[np.ndarray])->str:
    """Makes run-length encoding from 3x3 parts of image
        imgs: list[np.ndarray] - list of 9 images
    """
    temp =  " ".join(encode(np.concatenate([
        np.concatenate([imgs[i*3+j].reshape((1, SIZE, SIZE, 1)) for j in range(3)], axis=1)
        
     for i in range(3)], axis=2)[0].round()))
    return temp



model = UNET_model((SIZE,SIZE,3))
model.load_weights(MODEL_FILE)

gc.enable()


batch_size = 18 # you can make more if specs are powerfull (max value tested for kaggle)

# header
files = os.listdir(TEST_PATH)
with open(SUBMISSON_FILE, 'w') as file:
    file.write("ImageId,EncodedPixels")

for r in tqdm(range(0, len(files), batch_size)):
    batch_files = files[r:r+batch_size]
    batch_images = []
    for f in batch_files:
        temp = tfk.preprocessing.image.load_img(TEST_PATH+"/"+f)
        temp = tfk.preprocessing.image.img_to_array(temp)/255
        batch_images.extend([crop3x3(temp, 3*i+j) for i in range(3) for j in range(3)]) # cropping images and adding to batch
    out = model.predict_on_batch(np.array(batch_images))  # we process in batches for speed-up
    gc.collect()
    with open(SUBMISSON_FILE, 'a') as file:
        for o, p in zip(batch_files, [process(out[k*9:(k+1)*9]) for k in range(batch_size)]): # adding each encoded mask's data into file 
            file.write(f"\n{o},{p}") 



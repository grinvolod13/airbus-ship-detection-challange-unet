import keras as tfk
import numpy as np
import pandas as pd
import tensorflow as tf

from .crop import crop3x3, crop3x3_mask
from .encode_decode import decode
from .const import *


class TrainDataGenerator(tfk.utils.Sequence):

    def __init__(self, datapath ,batch_size, df_mask: pd.DataFrame):
        self.datapath = datapath
        self.batch_size = batch_size
        self.df =  df_mask.sample(frac=1)
        self.l = len(self.df)//batch_size

    def __len__(self):
        return self.l

    def on_epoch_end(self):
        pass

    def __getitem__(self, index):
        mask = np.empty((self.batch_size, SIZE , SIZE), np.float32)
        image = np.empty((self.batch_size, SIZE, SIZE, 3), np.float32)
        
        for b in range(self.batch_size):
            temp = tfk.preprocessing.image.load_img(self.datapath + '/' + self.df.iloc[index*self.batch_size+b]['ImageId'])
            temp = tfk.preprocessing.image.img_to_array(temp)/255
        
            mask[b], i = crop3x3_mask( # decoding mask from run-length format, and cropping part with maximum ship's area(№ i)
                decode(
                    self.df.iloc[index*self.batch_size+b]['EncodedPixels']
                )
            ) 
            image[b] = crop3x3(temp, i) # using corresponding to mask crop of image (№ i)
            
        return image, mask
    
    def getitem(self, index):
        return self.__getitem__(index)
    
def get_rle_dict(data):
    rle_dict: dict = {}
    for _, (image_id, rle_str) in data.iterrows():
        if isinstance(rle_str, float):
            rle_dict[image_id] = []
        elif image_id in rle_dict:
            rle_dict[image_id].append(rle_str)
        else:
            rle_dict[image_id] = [rle_str]
    return rle_dict

def get_datagenerator(datapath, batch_size, df_mask):
    d = TrainDataGenerator(datapath, df_mask=df_mask, batch_size=1)
    def _():
        for i in range(len(d)):
            x, y =  d.getitem(i)
            yield x[0], y[0]
    return tf.data.Dataset.from_generator(
        _, output_signature=(
            tf.TensorSpec(shape=(SIZE, SIZE, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(SIZE, SIZE), dtype=tf.float32),
        ), 
    )
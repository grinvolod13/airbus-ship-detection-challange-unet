import tensorflow as tf
from model import UNET_model
import keras as tfk
from src.const import SIZE, SIZE_FULL
from src.datagen import get_rle_dict, get_datagenerator
from sklearn.model_selection import train_test_split
import pandas as pd

empty, has_ships = 1000, 9000 # we will use part of dataset wich mostly contains ships (9000 images of ships + some of empty tiles)

dataset_csv_path = "/kaggle/input/airbus-ship-detection/train_ship_segmentations_v2.csv" # here is mine

df = pd.read_csv(dataset_csv_path)
df = pd.DataFrame(get_rle_dict(df).items(), columns=['ImageId', "EncodedPixels"])
df = pd.concat([df[df["EncodedPixels"].isin([[]])].sample(empty), df[~df["EncodedPixels"].isin([[]])].sample(has_ships)])
batch_size = 32
train_df, valid_df = train_test_split(df, test_size=0.2)
train = get_datagenerator("/kaggle/input/airbus-ship-detection/train_v2", 1, train_df)
valid = get_datagenerator("/kaggle/input/airbus-ship-detection/train_v2", 1, valid_df)
model = UNET_model((SIZE, SIZE, 3)) # model uses 256x256x3 images

train = train.cache().batch(batch_size).prefetch(1)
valid = valid.cache().batch(batch_size).prefetch(1)

from src.loss_metric import dice_score, BCE_dice
from src.callbacks import callback

model.compile(tf.keras.optimizers.Adam(0.001) , BCE_dice  , dice_score)
model.fit(train, validation_data=valid, batch_size = batch_size,epochs=8, verbose=1, callbacks=[callback], shuffle=True)

model.compile(tf.keras.optimizers.Adam(0.0005) , BCE_dice  , dice_score)
model.fit(train, validation_data=valid, batch_size = batch_size,epochs=8, verbose=1, callbacks=[callback], shuffle=True)

model.compile(tf.keras.optimizers.Adam(0.0001) , BCE_dice  , dice_score)
model.fit(train, validation_data=valid, batch_size = batch_size, epochs=8, verbose=1, callbacks=[callback], shuffle=True)

model.compile(tf.keras.optimizers.Adam(0.00001) , BCE_dice  , dice_score)
model.fit(train, validation_data=valid, batch_size = batch_size,epochs=8, verbose=1, callbacks=[callback], shuffle=True)

model.save_weights("./models/final.h5") # final model, but we also have weights saved by "best only" callback at models/checkpoints 

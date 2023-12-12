from model import UNET_model
import keras as tfk
from src.const import SIZE, SIZE_FULL
from src.datagen import TrainDataGenerator
from sklearn.model_selection import train_test_split
import pandas as pd

empty, has_ships = 1000, 9000 # we will use part of dataset wich mostly contains ships (9000 images of ships + some of empty tiles)

dataset_csv_path = "/kaggle/input/airbus-ship-detection/train_ship_segmentations_v2.csv" # here is mine

df = pd.read_csv(dataset_csv_path)
df = pd.concat([
    df[df["EncodedPixels"].isna()].sample(empty), # sample of images with no ship segmentation data (empty value in csv, NaN in pandas)
    df[~df["EncodedPixels"].isna()].sample(has_ships) # where isn't NaN
    ])

batch_size = 16
train_df, valid_df = train_test_split(df, test_size=0.2)

train = TrainDataGenerator("/kaggle/input/airbus-ship-detection/train_v2", batch_size, train_df)
valid = TrainDataGenerator("/kaggle/input/airbus-ship-detection/train_v2", batch_size, valid_df)


model = UNET_model((SIZE, SIZE, 3)) # model uses 256x256x3 images


from src.loss_metric import dice_score, BCE_dice
from src.callbacks import callback
# BCE_dice = binary focal cross-entropy + DiceLoss

model.compile(tfk.optimizers.Adam(0.0001), BCE_dice, dice_score)
model.fit(train,
          validation_data=valid,
          batch_size = batch_size,
          epochs=24,
          verbose=1,
          callbacks=[callback],
          )

model.compile(tfk.optimizers.Adam(0.00005), BCE_dice, dice_score)
model.fit(train,
          validation_data=valid,
          batch_size = batch_size,
          epochs=6,
          verbose=1,
          callbacks=[callback]
          )

model.compile(tfk.optimizers.Adam(0.00001), BCE_dice, dice_score)
model.fit(train,
          validation_data=valid,
          batch_size = batch_size,
          epochs=6,
          verbose=1,
          callbacks=[callback],
          )



model.save_weights("./models/final.h5") # final model, but we also have weights saved by "best only" callback at models/checkpoints 

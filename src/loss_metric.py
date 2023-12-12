import keras.backend as K


def dice_score(y_true, y_pred):
    return (2.0*K.sum(y_pred * y_true)+0.0001) / (K.sum(y_true)+ K.sum(y_pred)+0.0001)

def BFCE_dice(y_true, y_pred):
    return  K.binary_focal_crossentropy(y_true, y_pred)+  (1-dice_score(y_true, y_pred))*0.1

def BCE_dice(y_true, y_pred):
    return  K.binary_crossentropy(y_true, y_pred)+  (1-dice_score(y_true, y_pred))
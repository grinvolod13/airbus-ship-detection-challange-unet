import keras as tfk

callback = tfk.callbacks.ModelCheckpoint(
    "./models/checkpoints/model.epoch:{epoch:02d}-loss:{val_loss:.4f}-dice:{val_dice_score:.4f}.h5",
    "val_loss",
    save_best_only=True,
    save_weights_only=True,
    )
import os

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import KFold
from tqdm import tqdm

from logger import setup_logger

log = setup_logger(__name__)


def get_optimizer(name, learning_rate):
    tfk = tf.keras
    optimizers = {
        "adam": tfk.optimizers.Adam(learning_rate),
        "nadam": tfk.optimizers.Nadam(learning_rate),
        "adamax": tfk.optimizers.Adamax(learning_rate),
    }

    log.info(f"Using optimizer {optimizers[name]}")

    return optimizers[name]


def create_model(config, image_shape):
    # Create the model
    tfd = tfp.distributions
    tfk = tf.keras
    tfkl = tf.keras.layers

    # Define a Pixel CNN network
    dist = tfd.PixelCNN(
        image_shape=image_shape,
        num_resnet=1,
        num_hierarchies=3,
        num_filters=32,
        num_logistic_mix=5,
        dropout_p=config.dropout_rate,
    )

    # Define the model input
    image_input = tfkl.Input(shape=image_shape)

    # Define the log likelihood for the loss fn
    log_prob = dist.log_prob(image_input)

    # Define the model
    model = tfk.Model(inputs=image_input, outputs=log_prob)
    model.add_loss(-tf.reduce_mean(log_prob))

    # Compile and train the model
    current_optimizer = get_optimizer(config.optimizer, config.learning_rate)

    model.compile(optimizer=current_optimizer, metrics=[])

    return (model, dist)


def get_callbacks(config):
    # Create checkpoints of the model
    cp_callback = ModelCheckpoint(
        filepath=config.checkpoints + "cp-{epoch:04d}.ckpt",
        verbose=1,
        save_weights_only=True,
        save_freq=10 * config.batch_size,
    )

    # Early Stopping when loss stops improving
    early_stop = EarlyStopping(
        monitor="val_loss", min_delta=0, patience=config.patience, verbose=1, mode="min"
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=5, min_lr=0.001
    )

    return [cp_callback, early_stop, reduce_lr]


def train(data, val_ds, config, image_shape=(128, 128, 1)):
    log.info("Starting training...")

    # Create model
    model, dist = create_model(config, image_shape)

    # # history = model.fit
    history = model.fit(
        data,
        epochs=config.epochs,
        validation_data=val_ds,
        verbose=True,
        callbacks=get_callbacks(config),
    )

    log.info("Training done")

    # TODO: Save the data raw as well, so we can transform it later
    log.info("Saving history of loss...")
    hist_df = pd.DataFrame.from_dict(history.history)
    hist_df.to_csv(f"./{config.name}-history.csv")

    log.info("Plotting results...")
    plt.figure()
    epochs_range = range(config.epochs)
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    plt.plot(epochs_range, loss, label="training loss")
    plt.plot(epochs_range, val_loss, label="validation loss")
    plt.xlabel("Epochs")
    plt.legend(loc="lower right")
    plt.title("Training and Validation Loss")
    # plt.show()
    plt.savefig(f"./{config.name}-output.jpg")

    return (model, dist)


def save_image(image, image_counter, config):
    log.info(f"Saving image {image_counter} to disk...")
    if not config.output_dir.endswith("/"):
        config.output_dir += "/"
    if not os.path.isdir(config.output_dir):
        try:
            os.makedirs(config.output_dir)
        except Exception as e:
            log.warn(f"makedir exception: {e}")

    tf.keras.preprocessing.image.save_img(
        config.output_dir + f"output_{image_counter}.png", image
    )

    log.info("Saving done")


def predict(model, config):
    # Return n randomly sampled elements
    for idx in tqdm(range(config.output_number), desc="sample number "):
        save_image(model.sample(), idx, config)

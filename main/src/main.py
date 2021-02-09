import argparse

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import KFold
from tqdm import tqdm

import logger

log = None


def predict(model, num_of_samples=5):
    # Return n randomly sampled elements
    samples = []
    for _ in tqdm(range(num_of_samples), desc="sample number "):
        samples.append(model.sample())
    return samples


def create_model(config, image_shape):
    # Create the model
    tfd = tfp.distributions
    tfk = tf.keras
    tfkl = tf.keras.layers

    # Define a Pixel CNN network
    # TODO: Look at documentation of pixelCNN
    dist = tfd.PixelCNN(
        image_shape=image_shape,
        num_resnet=1,
        num_hierarchies=2,
        num_filters=32,
        num_logistic_mix=5,
        dropout_p=0.3,
    )

    # Define the model input
    # TODO: Understand
    image_input = tfkl.Input(shape=image_shape)

    # Define the log likelihood for the loss fn
    # TODO: Understand
    log_prob = dist.log_prob(image_input)

    # Define the model
    # TODO: Understand
    model = tfk.Model(inputs=image_input, outputs=log_prob)
    model.add_loss(-tf.reduce_mean(log_prob))

    # Compile and train the model
    # TODO: Understand
    model.compile(optimizer=tfk.optimizers.Adamax(0.001), metrics=["accuracy"])

    return (model, dist)


def get_callbacks():
    # Create checkpoints of the model
    cp_callback = ModelCheckpoint(
        filepath=config.checkpoints + "cp-{epoch:04d}.ckpt",
        verbose=1,
        save_weights_only=True,
        save_freq=10 * config.batch_size,
    )

    # Early Stopping when loss stops improving
    early_stop = EarlyStopping(
        monitor="loss", min_delta=0, patience=config.patience, verbose=1, mode="min"
    )

    return [cp_callback, early_stop]


def train(data, config, image_shape=(128, 128, 1)):
    log.info("Starting training...")

    # n_split = 3

    # for train_index, test_index in KFold(n_split).split(X):
    #     x_train, x_test = X[train_index], X[test_index]
    #     y_train, y_test = Y[train_index], Y[test_index]

    #     model, dist = create_model(config, image_shape)
    #     model.fit(x_train, y_train, epochs=config.epochs)

    #     log.debug("Model evaluation {}".format(model.evaluate(x_test, y_test)))

    # Create model
    model, dist = create_model(config, image_shape)

    log.info(data)

    # TODO: Understand
    history = model.fit(
        data,
        epochs=100,
        verbose=True,  # callbacks=get_callbacks()
    )
    # history = model.fit(data, epochs=config.epochs, validation_split=0.2,
    #           verbose=True , callbacks=get_callbacks())

    # Save the model
    # model.save("{}/model/saved_model".format(config.output_dir))

    log.debug(dist)
    log.debug(model)
    log.info("Training done")

    log.info("Plotting results...")
    plt.figure()
    epochs_range = range(config.epochs)
    loss = history.history["loss"]
    # val_loss = history.history["val_loss"]
    plt.plot(epochs_range, loss, label="training loss")
    # plt.plot(epochs_range, val_loss, label="validation loss")
    plt.xlabel("Epochs")
    plt.legend(loc="lower right")
    plt.title("Training and Validation Loss")
    plt.show()

    return (model, dist)


def main(config):

    log.info("Starting...")

    log.info("Loading images...")
    data = tf.keras.preprocessing.image_dataset_from_directory(
        config.input_dir,
        seed=config.seed,
        color_mode="grayscale",
        batch_size=config.batch_size,
        image_size=(128, 128),
    )
    # data = tf.keras.preprocessing.image_dataset_from_directory(
    #     config.input_dir, seed=config.seed, color_mode="grayscale", batch_size=config.batch_size, image_size=(128, 128)
    #     , validation_split=0.2, subset="validation")
    log.debug(data)
    log.info("Loading images done")

    # log.info("Preprocessing...")
    # input_data = data.as_numpy_iterator()
    # X_train, X_test, y_train, y_test = train_test_split(
    #     input_data, test_size=0.25, random_state=0
    # )

    if config.training:
        model, dist = train(data, config)
    else:
        log.info("Loading model...")
        # Load model
        latest = tf.train.latest_checkpoint(config.checkpoints)

        # Create a new model instance
        model, dist = create_model(config, (128, 128, 1))

        # Load the params back into the model
        model.load_weights(latest).expect_partial()

        # TODO
        # tf.keras.utils.plot_model(model, "model_graph.png")
        log.info("Loading done")

    log.info("Predicting...")
    prediction = predict(dist, config.output_number)
    log.debug(prediction)
    log.info("Prediction done...")

    log.info("Saving images to disk...")
    image_counter = 0
    for image in prediction:
        tf.keras.preprocessing.image.save_img(
            config.output_dir + f"output_{image_counter}.png", image
        )
        image_counter += 1
    log.info("Saving done")

    log.info("Process complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input_dir", help="input directory where all images are found")
    parser.add_argument(
        "output_dir",
        help="output directory where all newly generated images will be placed",
    )
    parser.add_argument(
        "--output_number",
        help="Depends on how many images should be sampled from the resulting model distribution",
        default=2,
        type=int,
    )
    parser.add_argument(
        "--training", help="flag: Set to true when training", action="store_true"
    )
    parser.add_argument(
        "--epochs", help="Set the number of epochs", default=10, type=int
    )
    parser.add_argument(
        "--batch_size",
        help="Sets the size of the batch, ideally this divides nicely through the number of images",
        default=16,
        type=int,
    )
    parser.add_argument(
        "--checkpoints", help="Set the path to save checkpoints", default="."
    )
    parser.add_argument(
        "--seed",
        help="value: Set a seed for the determined randomness",
        default=None,
        type=int,
    )
    parser.add_argument(
        "--model",
        help="path to saved keras distribution model",
        default="./models/cnn_save",
    )
    parser.add_argument(
        "--log_level",
        help="Specifies the level of precision for the logger",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
    )
    parser.add_argument(
        "--patience",
        help="patience level of early stopping",
        type=int,
        default=10,
    )

    config = parser.parse_args()

    print("Program will run with following parameters:")
    print(config)

    # Setup global logger
    log = logger.setup_logger(__name__, level=config.log_level)

    # Run the model with the specified parameters
    main(config)

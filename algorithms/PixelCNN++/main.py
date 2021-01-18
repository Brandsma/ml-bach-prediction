import tensorflow_probability as tfp
import tensorflow_datasets as tfds
import tensorflow as tf
import logger
import argparse

log = None


def predict(model, num_of_sample=5):
    # Return n randomly sampled elements
    return model.sample(num_of_sample)


def train(data, epochs=10, image_shape=(128, 128, 1)):
    log.info("Starting training...")
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
        dropout_p=.3,
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
    model.compile(
        optimizer=tfk.optimizers.Adam(.001),
        metrics=[])

    # TODO: Understand
    # TODO: Save the model at some point
    model.fit(data, epochs=epochs, verbose=True)

    log.debug(dist)
    log.info("Training done")

    return dist


def main(config):

    log.info("Starting...")

    log.info("Loading images...")
    data = tf.keras.preprocessing.image_dataset_from_directory(
        config.input_dir, seed=config.seed, color_mode="grayscale", batch_size=1, image_size=(128, 128))
    log.debug(data)
    log.info("Loading images done")

    if config.training:
        model = train(data, epochs=config.epochs)

        log.info("Saving model...")
        # TODO: Save modelsave_
        #tf.saved_model.save(model, config.model)
        log.info("Saving done")
    else:
        # TODO: Load model
        pass
        # model = None

    log.info("Predicting...")
    prediction = predict(model, num_of_sample=1)
    log.debug(prediction)
    log.info("Prediction done...")

    log.info("Saving images to disk...")
    image_counter = 0
    for image in prediction:
        tf.keras.preprocessing.save_img(
            config.output_dir + f"output_{image_counter}.png", image)
    log.info("Saving done")

    log.info("Process complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "input_dir", help="input directory where all images are found")
    parser.add_argument(
        "output_dir", help="output directory where all newly generated images will be placed")
    parser.add_argument(
        "--output_number", help="Depends on how many images should be sampled from the resulting model distribution", default=1, type=int)
    parser.add_argument(
        "--training", help="flag: Set to true when training", action="store_true")
    parser.add_argument(
        "--epochs", help="Set the number of epochs", default=10, type=int)
    parser.add_argument(
        "--seed", help="value: Set a seed for the determined randomness", default=None, type=int)
    parser.add_argument(
        "--model", help="path to saved keras distribution model", default="./models/cnn_save")
    parser.add_argument("--log_level", help="Specifies the level of precision for the logger",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], default="INFO")

    config = parser.parse_args()

    # Setup global logger
    log = logger.setup_logger(__name__, level=config.log_level)

    # Run the model with the specified parameters
    main(config)

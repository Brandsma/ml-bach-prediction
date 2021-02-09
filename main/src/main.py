import argparse
import os

import tensorflow as tf

import logger
from model import create_model, predict, train

# Setup global logger
log = logger.setup_logger(__name__)


def create_dataset(config):
    log.info("Loading dataset...")

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        config.input_dir,
        seed=config.seed,
        color_mode="grayscale",
        batch_size=config.batch_size,
        validation_split=0.2,
        subset="training",
        # TODO: Change to proper size
        image_size=(128, 128),
    )

    validation_ds = tf.keras.preprocessing.image_dataset_from_directory(
        config.input_dir,
        seed=config.seed,
        color_mode="grayscale",
        batch_size=config.batch_size,
        validation_split=0.2,
        subset="validation",
        # TODO: Change to proper size
        image_size=(128, 128),
    )

    log.info("Loading dataset done")

    return (train_ds, validation_ds)


def run_model(train_ds, validation_ds, config):
    if config.training:
        model, dist = train(train_ds, config)
        print(f"Model evaluation: {model.evaluate(validation_ds)}")
    else:
        log.info("Loading model...")
        # Load model
        latest = tf.train.latest_checkpoint(config.checkpoints)

        # Create a new model instance
        # TODO: Use proper image size
        model, dist = create_model(config, (128, 128, 1))

        # Load the params back into the model
        model.load_weights(latest).expect_partial()

        log.info("Loading done")

    log.info("Predicting...")

    prediction = predict(dist, config.output_number)
    log.debug(prediction)

    log.info("Prediction done...")


def save_images(prediction, config):
    log.info("Saving images to disk...")
    if not config.output_dir.endswith("/"):
        config.output_dir += "/"
    if os.path.isdir(config.output_dir):
        os.makedirs(config.output_dir)

    image_counter = 0
    for image in prediction:
        tf.keras.preprocessing.image.save_img(
            config.output_dir + f"output_{image_counter}.png", image
        )
        image_counter += 1

    log.info("Saving done")


def main():
    # Extract config from arguments
    config = setup_argument_parser()

    log.info("Starting...")

    log.info("Program will run with following parameters:")
    log.info(config)

    train_ds, validation_ds = create_dataset(config)

    prediction = run_model(train_ds, validation_ds, config)

    save_images(prediction, config)

    log.info("  Done  ")


def setup_argument_parser():
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
        default=42,
        type=int,
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
    parser.add_argument(
        "--learning_rate",
        help="learning rate of the optimizer, range is 0-1",
        type=float,
        default=0.001,
    )
    parser.add_argument(
        "--optimizer",
        help="Determines the optimizer to use for gradient descent",
        type=str,
        default="adam",
        choices=["adam", "nadam", "adamax"],
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()

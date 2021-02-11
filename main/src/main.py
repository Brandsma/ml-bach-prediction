import argparse

import tensorflow as tf
from tensorflow.keras import layers

import logger
from model import create_model, predict, train

# Setup global logger
log = logger.setup_logger(__name__)


def get_image_size(image_size_file):
    y_length = 0
    with open(image_size_file, "r") as f:
        content = f.readlines()
        y_length = int(content[1]) - int(content[0]) + 1

    return ((128, 128, 1), (128, 128))
    # return ((128, y_length, 1), (128, y_length))


def create_dataset(config):
    log.info("Loading dataset...")

    if config.image_size_file is not None:
        image_size, simplified_image_size = get_image_size(config.image_size_file)
    else:
        image_size, simplified_image_size = (128, 128, 1)

    log.info(f"Input images have dimension {image_size}")

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        config.input_dir,
        seed=config.seed,
        color_mode="grayscale",
        batch_size=config.batch_size,
        validation_split=0.2,
        subset="training",
        # TODO: Change to proper size
        image_size=simplified_image_size,
    )

    validation_ds = tf.keras.preprocessing.image_dataset_from_directory(
        config.input_dir,
        seed=config.seed,
        color_mode="grayscale",
        batch_size=config.batch_size,
        validation_split=0.2,
        subset="validation",
        # TODO: Change to proper size
        image_size=simplified_image_size,
    )

    # Normalize the data between 0 and 1
    normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(
        1.0 / 255
    )
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    validation_ds = validation_ds.map(lambda x, y: (normalization_layer(x), y))

    # Add a prefetch mechanic so producers and consumers can work at the same time
    train_ds = train_ds.cache().prefetch(2)
    validation_ds = validation_ds.cache().prefetch(2)

    log.info("Loading dataset done")

    return (train_ds, validation_ds, image_size)


def run_model(train_ds, validation_ds, config, image_size=(128, 128, 1)):

    if config.training:
        model, dist = train(train_ds, validation_ds, config, image_shape=image_size)
        with open(f"./{config.name}-evaluation.txt", "w") as f:
            log.info(f"Writing validation loss to ./{config.name}-evaluation.txt...")
            f.write(f"loss:\n{model.evaluate(validation_ds)}")
    else:
        log.info("Loading model...")
        # Load model
        latest = tf.train.latest_checkpoint(config.checkpoints)
        log.info(latest)

        # Create a new model instance
        # TODO: Use proper image size
        model, dist = create_model(config, image_size)

        # Load the params back into the model
        model.load_weights(latest).expect_partial()

        with open("./evaluation.txt", "w") as f:
            log.info("Writing validation loss to ./evaluation.txt...")
            f.write(f"loss:\n{model.evaluate(validation_ds)}")

        log.info("Loading done")

    log.info("Predicting...")

    predict(dist, config)

    log.info("Prediction done...")


def main():
    # Extract config from arguments
    config = setup_argument_parser()

    log.info("Starting...")

    log.info("Program will run with following parameters:")
    log.info(config)

    ds, val_ds, image_size = create_dataset(config)

    _ = run_model(ds, val_ds, config, image_size=image_size)

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
        "--image_size_file",
        help="Specification file for the highest and lowest note of the input images, default size is (128,128,1)",
        default=None,
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
        "--kfold_splits",
        help="Number of splits in the kfold cross-validation",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--learning_rate",
        help="learning rate of the optimizer, range is 0-1",
        type=float,
        default=0.005,
    )
    parser.add_argument(
        "--name",
        help="Name of the program right now",
        type=str,
        default="default_name_for_file",
    )
    parser.add_argument(
        "--dropout_rate",
        help="dropout rate of the pixelCNN distribution",
        type=float,
        default=0.5,
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

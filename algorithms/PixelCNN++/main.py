import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp


def image_preprocess(x):
    # TODO: Transform to greyscale/binary?
    x['image'] = tf.cast(x['image'], tf.float32)
    return (x['image'],)  # (input, output) of the model


def train(data, epochs=10, image_shape=(28, 28, 1)):
    tfd = tfp.distributions
    tfk = tf.keras
    tfkl = tf.keras.layers

    # data = input_data.map(image_preprocess)

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

    return dist


# @tf.function(experimental_relax_shapes=True)
def predict(model, num_of_sample=5):
    # Return 5 randomly sampled elements
    return model.sample(num_of_sample)


if __name__ == "__main__":
    print("Starting...")
    directory = "./input"

    print("Loading images...")
    data = tf.keras.preprocessing.image_dataset_from_directory(
        directory, seed=1337, color_mode="grayscale", batch_size=1, image_size=(128, 128))
    print(data)

    print("Starting training...")
    image_shape = (128, 128, 1)
    model = train(data, epochs=1, image_shape=image_shape)
    print(model)

    print("Predicting...")
    # print(predict(model))
    print("Done.")

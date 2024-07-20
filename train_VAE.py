import tensorflow as tf
from model import *
from tensorflow.keras import losses, optimizers
from tensorflow.keras.preprocessing import image_dataset_from_directory
from preprocessing_functions import *



if __name__ == '__main__':

    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        print(f"GPU Device: {gpu}")
        tf.config.experimental.set_memory_growth(gpu, True)

    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    # Define parameters
    root_dir = r"D:\\VIT Material\\VIT material\\Hepatoma Research Project\\Histopathology-Images"
    batch_size = 32
    img_size = 1600
    prefetch_size = 64

    # Create a dataset
    dataset = image_dataset_from_directory(
        root_dir,
        batch_size=batch_size,
        image_size=(img_size, img_size),
        shuffle=True
    )
    dataset = dataset.map(preprocess_image)
    dataset = dataset.cache()
    dataset = dataset.shuffle(300) # Shuffle Seed Values
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(prefetch_size) # Reduces the possibility of Bottlenecking


    with strategy.scope():
        encoder1 = Encoder1()
        encoder2 = Encoder2()
        decoder1 = Decoder1()
        decoder2 = Decoder2()
        vae = VAE(encoder1, encoder2, decoder1, decoder2)

    print(vae.summary())

    optimizer = optimizers.Adam()

    def vae_loss(inputs, reconstructed, z_mean, z_log_var, w1=1.0, w2=0.01):
        reconstruction_loss = losses.mean_squared_error(tf.keras.backend.flatten(inputs), tf.keras.backend.flatten(reconstructed))
        reconstruction_loss *= 1600 * 1600
        kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
        total_loss = w1 * reconstruction_loss + w2 * kl_loss
        return total_loss

    @tf.function
    def train_step(inputs):
        print("Entered training step")
        with tf.GradientTape() as tape:
            reconstructed, z_mean, z_log_var = vae(inputs)
            loss = vae_loss(inputs, reconstructed, z_mean, z_log_var)
        gradients = tape.gradient(loss, vae.trainable_variables)
        optimizer.apply_gradients(zip(gradients, vae.trainable_variables))
        return loss

    # # Training Loop
    # epochs = 50
    # for epoch in range(epochs):
    #     print(f"Starting epoch {epoch + 1}")
    #     for batch in dataset:  # Assuming dataset is a tf.data.Dataset object
    #         loss = train_step(batch)
    #     print(f'Epoch {epoch + 1}, Loss: {loss.numpy()}')

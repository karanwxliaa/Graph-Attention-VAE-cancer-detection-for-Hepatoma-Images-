import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import layers, Model, losses, optimizers

def preprocess_image(image, label):
    image = tf.cast(image, tf.float32) / 255.0  # Rescale to [0, 1]
    return image, label

class Encoder1(Model):
    def __init__(self):
        super(Encoder1, self).__init__()
        self.conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')
        self.pool1 = layers.MaxPooling2D((2, 2), padding='same')
        self.conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')
        self.pool2 = layers.MaxPooling2D((2, 2), padding='same')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(inputs)
        x = self.pool2(x)
        return x

class Encoder2(Model):
    def __init__(self):
        super(Encoder2, self).__init__()
        self.conv3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')
        self.pool3 = layers.MaxPooling2D((2, 2), padding='same')
        self.conv4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')
        self.pool4 = layers.MaxPooling2D((2, 2), padding='same')
        self.conv_mean = layers.Conv2D(128, (3, 3), activation='relu', padding='same')
        self.conv_log_var = layers.Conv2D(128, (3, 3), activation='relu', padding='same')

    def call(self, inputs):
        x = self.conv3(inputs)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.pool4(x)
        z_mean = self.conv_mean(x)
        z_log_var = self.conv_log_var(x)
        return z_mean, z_log_var

class Decoder1(Model):
    def __init__(self):
        super(Decoder1, self).__init__()
        self.deconv1 = layers.Conv2DTranspose(256, (3, 3), activation='relu', padding='same')
        self.upsample1 = layers.UpSampling2D((2, 2))
        self.deconv2 = layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same')
        self.upsample2 = layers.UpSampling2D((2, 2))

    def call(self, inputs):
        x = self.deconv1(inputs)
        x = self.upsample1(x)
        x = self.deconv2(x)
        x = self.upsample2(x)
        return x

class Decoder2(Model):
    def __init__(self):
        super(Decoder2, self).__init__()
        self.deconv3 = layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')
        self.upsample3 = layers.UpSampling2D((2, 2))
        self.deconv4 = layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same')
        self.upsample4 = layers.UpSampling2D((2, 2))
        self.output_layer = layers.Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same')

    def call(self, inputs):
        x = self.deconv3(inputs)
        x = self.upsample3(x)
        x = self.deconv4(x)
        x = self.upsample4(x)
        x = self.output_layer(x)
        return x

class VAE(Model):
    def __init__(self, encoder1, encoder2, decoder1, decoder2):
        super(VAE, self).__init__()
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        self.decoder1 = decoder1
        self.decoder2 = decoder2

    def call(self, inputs):
        x = self.encoder1(inputs)
        z_mean, z_log_var = self.encoder2(x)
        z = self.sampling([z_mean, z_log_var])
        x = self.decoder1(z)
        reconstructed = self.decoder2(x)
        return reconstructed, z_mean, z_log_var

    def sampling(self, args):
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        epsilon = tf.keras.backend.random_normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

if __name__ == '__main__':

    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        print(f"GPU Device: {gpu}")
        tf.config.experimental.set_memory_growth(gpu, True)

    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    # Define parameters
    root_dir = r"D:\\VIT Material\\VIT material\\Hepatoma Research Project\\Histopathology-Images"
    batch_size = 4
    img_size = 1600
    prefetch_size = 2

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
    
    # Build the model by calling it on a batch of data
    for images, _ in dataset.take(1):
        vae(images)
    
    print(vae.summary())

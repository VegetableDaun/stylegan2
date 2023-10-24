import tensorflow as tf

from stylegan2_discriminator import StyleGan2Discriminator
from stylegan2_generator import StyleGan2Generator

num_classes = 10


class StyleGan2(tf.keras.Model):
    """ 
    StyleGan2 config f for tensorflow 2.x 
    """

    def __init__(self, resolution=1024, weights=None, impl='cuda', gpu=True, **kwargs):
        """
        Parameters
        ----------
        resolution : int, optional
            Resolution output of the synthesis network, will be parsed
            to the floor integer power of 2. 
            The default is 1024.
        weights : string, optional
            weights name in weights dir to be loaded. The default is None.
        impl : str, optional
            Wether to run some convolutions in custom tensorflow
             operations or cuda operations. 'ref' and 'cuda' available.
            The default is 'cuda'.
        gpu : boolean, optional
            Wether to use gpu. The default is True.
        """
        super(StyleGan2, self).__init__(**kwargs)

        self.loss_fn = None
        self.g_optimizer = None
        self.d_optimizer = None

        self.resolution = resolution
        if weights is not None:
            self.__adjust_resolution(weights)
        self.generator = StyleGan2Generator(resolution=self.resolution, weights=weights,
                                            impl=impl, gpu=gpu, name='Generator')
        self.discriminator = StyleGan2Discriminator(resolution=self.resolution, weights=weights,
                                                    impl=impl, gpu=gpu, name='Discriminator')

    def call(self, latent_vector):
        """
        Parameters
        ----------
        latent_vector : latent vector z of size [batch, 512].

        Returns
        -------
        score : output of the discriminator. 
        """
        img = self.generator(latent_vector)
        score = self.discriminator(img)

        return score

    @property
    def metrics(self):
        return [self.gen_loss_tracker, self.disc_loss_tracker]

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    @tf.function
    def train_step(self, data):
        # Unpack the data.
        real_images, one_hot_labels = data

        # Add dummy dimensions to the labels so that they can be concatenated with
        # the images. This is for the discriminator.
        image_one_hot_labels = one_hot_labels[:, :, None, None]
        image_one_hot_labels = tf.repeat(
            image_one_hot_labels, repeats=[self.resolution * self.resolution]
        )
        image_one_hot_labels = tf.reshape(
            image_one_hot_labels, (-1, self.resolution, self.resolution, num_classes)
        )

        # Sample random points in the latent space and concatenate the labels.
        # This is for the generator.
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        random_vector_labels = tf.concat(
            [random_latent_vectors, one_hot_labels], axis=1
        )

        # Decode the noise (guided by labels) to fake images.
        generated_images = self.generator(random_vector_labels)

        # Combine them with real images. Note that we are concatenating the labels
        # with these images here.
        fake_image_and_labels = tf.concat([generated_images, image_one_hot_labels], -1)
        real_image_and_labels = tf.concat([real_images, image_one_hot_labels], -1)
        combined_images = tf.concat(
            [fake_image_and_labels, real_image_and_labels], axis=0
        )

        # Assemble labels discriminating real from fake images.
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )

        # Train the discriminator.
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        # Sample random points in the latent space.
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        random_vector_labels = tf.concat(
            [random_latent_vectors, one_hot_labels], axis=1
        )

        # Assemble labels that say "all real images".
        misleading_labels = tf.zeros((batch_size, 1))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
            fake_images = self.generator(random_vector_labels)
            fake_image_and_labels = tf.concat([fake_images, image_one_hot_labels], -1)
            predictions = self.discriminator(fake_image_and_labels)
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # Monitor loss.
        self.gen_loss_tracker.update_state(g_loss)
        self.disc_loss_tracker.update_state(d_loss)
        return {
            "g_loss": self.gen_loss_tracker.result(),
            "d_loss": self.disc_loss_tracker.result(),
        }

    def __adjust_resolution(self, weights_name):
        """
        Adjust resolution of the synthesis network output. 
        
        Parameters
        ----------
        weights_name : name of the weights
        """
        if weights_name == 'ffhq':
            self.resolution = 1024
        elif weights_name == 'car':
            self.resolution = 512
        elif weights_name in ['cat', 'church', 'horse']:
            self.resolution = 256

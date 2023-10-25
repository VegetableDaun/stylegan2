import tensorflow as tf
import keras

from stylegan2_discriminator import StyleGan2Discriminator
from stylegan2_generator import StyleGan2Generator
from config_GAN import latent_dim, num_classes

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
        self.latent_dim = latent_dim
        self.loss_weights = {"gradient_penalty": 10, "drift": 0.001}

        self.gen_loss_tracker = keras.metrics.Mean(name="generator_loss")
        self.disc_loss_tracker = keras.metrics.Mean(name="discriminator_loss")

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

    # @property
    # def metrics(self):
    #     return [self.gen_loss_tracker, self.disc_loss_tracker]

    # def compile(self, d_optimizer, g_optimizer, loss_fn):
    #     super().compile()
    #     self.d_optimizer = d_optimizer
    #     self.g_optimizer = g_optimizer
    #     self.loss_fn = loss_fn
    #
    # @tf.function
    # def train_step(self, data):
    #     # Unpack the data.
    #     real_images, one_hot_labels = data
    #
    #     # Add dummy dimensions to the labels so that they can be concatenated with
    #     # the images. This is for the discriminator.
    #     image_one_hot_labels = one_hot_labels[:, :, None, None]
    #     image_one_hot_labels = tf.repeat(
    #         image_one_hot_labels, repeats=[self.resolution * self.resolution]
    #     )
    #     image_one_hot_labels = tf.reshape(
    #         image_one_hot_labels, (-1, self.resolution, self.resolution, num_classes)
    #     )
    #
    #     # Sample random points in the latent space and concatenate the labels.
    #     # This is for the generator.
    #     batch_size = tf.shape(real_images)[0]
    #     random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
    #     random_vector_labels = tf.concat(
    #         [random_latent_vectors, one_hot_labels], axis=1
    #     )
    #
    #     # Decode the noise (guided by labels) to fake images.
    #     generated_images = self.generator(random_vector_labels)
    #
    #     # Combine them with real images. Note that we are concatenating the labels
    #     # with these images here.
    #     fake_image_and_labels = tf.concat([generated_images, image_one_hot_labels], -1)
    #     real_image_and_labels = tf.concat([real_images, image_one_hot_labels], -1)
    #     combined_images = tf.concat(
    #         [fake_image_and_labels, real_image_and_labels], axis=0
    #     )
    #
    #     # Assemble labels discriminating real from fake images.
    #     labels = tf.concat(
    #         [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
    #     )
    #
    #     # Train the discriminator.
    #     with tf.GradientTape() as tape:
    #         predictions = self.discriminator(combined_images)
    #         d_loss = self.loss_fn(labels, predictions)
    #     grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
    #     self.d_optimizer.apply_gradients(
    #         zip(grads, self.discriminator.trainable_weights)
    #     )
    #
    #     # Sample random points in the latent space.
    #     random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
    #     random_vector_labels = tf.concat(
    #         [random_latent_vectors, one_hot_labels], axis=1
    #     )
    #
    #     # Assemble labels that say "all real images".
    #     misleading_labels = tf.zeros((batch_size, 1))
    #
    #     # Train the generator (note that we should *not* update the weights
    #     # of the discriminator)!
    #     with tf.GradientTape() as tape:
    #         fake_images = self.generator(random_vector_labels)
    #         fake_image_and_labels = tf.concat([fake_images, image_one_hot_labels], -1)
    #         predictions = self.discriminator(fake_image_and_labels)
    #         g_loss = self.loss_fn(misleading_labels, predictions)
    #     grads = tape.gradient(g_loss, self.generator.trainable_weights)
    #     self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
    #
    #     # Monitor loss.
    #     self.gen_loss_tracker.update_state(g_loss)
    #     self.disc_loss_tracker.update_state(d_loss)
    #     return {
    #         "g_loss": self.gen_loss_tracker.result(),
    #         "d_loss": self.disc_loss_tracker.result(),
    #     }

    @staticmethod
    def wasserstein_loss(y_true, y_pred):
        return -tf.reduce_mean(y_true * y_pred)

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    def compile(self, d_optimizer, g_optimizer, *args, **kwargs):
        self.loss_weights = kwargs.pop("loss_weights", self.loss_weights)
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer

        self.d_loss_metric = keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")

        super().compile(*args, **kwargs)

    def gradient_loss(self, grad):
        loss = tf.square(grad)
        loss = tf.reduce_sum(loss, axis=tf.range(1, tf.size(tf.shape(loss))))
        loss = tf.sqrt(loss)
        loss = tf.reduce_mean(tf.square(loss - 1))
        return loss

    @tf.function
    def train_step(self, data):
        real_images, one_hot_labels = data

        batch_size = tf.shape(real_images)[0]
        real_labels = tf.ones(batch_size)
        fake_labels = -tf.ones(batch_size)

        noise = tf.random.normal(shape=(batch_size, self.latent_dim))

        # generator
        with tf.GradientTape() as g_tape:
            fake_images = self.generator(noise)
            pred_fake = self.discriminator(fake_images)
            g_loss = self.wasserstein_loss(real_labels, pred_fake)

            trainable_weights = (self.generator.mapping_network.trainable_weights
                                 + self.generator.synthesis_network.trainable_weights)
            # trainable_weights = self.generator.trainable_weights
            gradients = g_tape.gradient(g_loss, trainable_weights)
            self.g_optimizer.apply_gradients(zip(gradients, trainable_weights))

        # discriminator
        with tf.GradientTape() as gradient_tape, tf.GradientTape() as total_tape:
            # forward pass
            pred_fake = self.discriminator(fake_images)
            real_images = tf.transpose(real_images, [0, 2, 3, 1])
            pred_real = self.discriminator(real_images)

            epsilon = tf.random.uniform((batch_size, 1, 1, 1))
            interpolates = epsilon * real_images + (1 - epsilon) * fake_images
            gradient_tape.watch(interpolates)
            pred_fake_grad = self.discriminator(interpolates)

            # calculate losses
            loss_fake = self.wasserstein_loss(fake_labels, pred_fake)
            loss_real = self.wasserstein_loss(real_labels, pred_real)
            loss_fake_grad = self.wasserstein_loss(fake_labels, pred_fake_grad)

            # gradient penalty
            gradients_fake = gradient_tape.gradient(loss_fake_grad, [interpolates])
            gradient_penalty = self.loss_weights["gradient_penalty"] * self.gradient_loss(gradients_fake)

            # drift loss
            all_pred = tf.concat([pred_fake, pred_real], axis=0)
            drift_loss = self.loss_weights["drift"] * tf.reduce_mean(all_pred ** 2)

            d_loss = loss_fake + loss_real + gradient_penalty + drift_loss

            gradients = total_tape.gradient(
                d_loss, self.discriminator.trainable_weights
            )
            self.d_optimizer.apply_gradients(
                zip(gradients, self.discriminator.trainable_weights)
            )

        # Update metrics
        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)

        return {
            "d_loss": self.d_loss_metric.result(),
            "g_loss": self.g_loss_metric.result(),
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

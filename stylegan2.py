import keras
import tensorflow as tf

from config_GAN import latent_dim
from stylegan2_discriminator import StyleGan2Discriminator
from stylegan2_generator import StyleGan2Generator


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

        self.g_optimizer = None
        self.d_optimizer = None
        self.latent_dim = latent_dim
        self.loss_weights = {"gradient_penalty": 10, "drift": 0.001}

        self.resolution = resolution
        if weights is not None:
            self.__adjust_resolution(weights)
        self.generator = StyleGan2Generator(resolution=self.resolution, weights=weights,
                                            impl=impl, gpu=gpu, name='Generator')
        self.discriminator = StyleGan2Discriminator(resolution=self.resolution, weights=weights,
                                                    impl=impl, gpu=gpu, name='Discriminator')

    def call(self, latent_vector=1, c=None):
        """
        Parameters
        ----------
        latent_vector : latent vector z of size [batch, 512].

        Returns
        -------
        score : output of the discriminator. 
        """

        assert c is None, "Use conditional"

        img = self.generator(latent_vector, lambda_t=latent_vector, c=c)
        score = self.discriminator(img, c=c)

        return score

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    def compile(self, d_optimizer, g_optimizer, T_s, T_e, epoch=1, *args, **kwargs):
        self.loss_weights = kwargs.pop("loss_weights", self.loss_weights)
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer

        self.T_s = T_s
        self.T_e = T_e
        self.epoch = epoch

        self.d_loss_metric = keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")

        super().compile(*args, **kwargs)

    def lambda_t(self, epoch):
        return min([max([(epoch - self.T_s) / (self.T_e - self.T_s), 0]), 1])

    @staticmethod
    def wasserstein_loss(y_true, y_pred):
        return -tf.reduce_mean(y_true * y_pred)

    def gradient_loss(self, grad):
        loss = tf.square(grad)
        loss = tf.reduce_sum(loss, axis=tf.range(1, tf.size(tf.shape(loss))))
        loss = tf.sqrt(loss)
        loss = tf.reduce_mean(tf.square(loss - 1))
        return loss

    @tf.function
    def train_step(self, data):
        real_images, one_hot_labels = data
        real_images = tf.transpose(real_images, [0, 3, 1, 2])

        batch_size = tf.shape(real_images)[0]
        real_labels = tf.ones(batch_size)
        fake_labels = -tf.ones(batch_size)

        noise = tf.random.normal(shape=(batch_size, self.latent_dim))

        # generator
        with tf.GradientTape() as g_tape:
            fake_images = self.generator(noise, lambda_t=self.lambda_t(self.epoch), c=one_hot_labels)
            pred_fake = self.discriminator(fake_images, c=one_hot_labels)
            g_loss = (self.wasserstein_loss(real_labels, pred_fake[0])
                      + self.lambda_t(self.epoch) * pred_fake[1])

            trainable_weights = (self.generator.mapping_network.trainable_weights
                                 + self.generator.synthesis_network.trainable_weights)
            gradients = g_tape.gradient(g_loss, trainable_weights)
            self.g_optimizer.apply_gradients(zip(gradients, trainable_weights))

        # discriminator
        with tf.GradientTape() as gradient_tape, tf.GradientTape() as total_tape:
            # forward pass
            pred_fake = self.discriminator(fake_images, c=one_hot_labels)
            pred_real = self.discriminator(real_images, c=one_hot_labels)

            epsilon = tf.random.uniform((batch_size, 1, 1, 1))
            interpolates = epsilon * real_images + (1 - epsilon) * fake_images
            gradient_tape.watch(interpolates)
            pred_fake_grad = self.discriminator(interpolates, c=one_hot_labels)

            # calculate losses
            loss_fake = (self.wasserstein_loss(fake_labels, pred_fake[0])
                             + self.lambda_t(self.epoch) * pred_fake[1])

            loss_real = (self.wasserstein_loss(real_labels, pred_real[0])
                             + self.lambda_t(self.epoch) * pred_real[1])

            loss_fake_grad = (self.wasserstein_loss(real_labels, pred_fake_grad[0])
                             + self.lambda_t(self.epoch) * pred_fake_grad[1])


            # gradient penalty
            gradients_fake = gradient_tape.gradient(loss_fake_grad, [interpolates])
            gradient_penalty = self.loss_weights["gradient_penalty"] * self.gradient_loss(gradients_fake)

            # drift loss
            new_pred_fake = pred_fake[0] + self.lambda_t(self.epoch) * pred_fake[1]
            new_pred_real = pred_real[0] + self.lambda_t(self.epoch) * pred_real[1]
            all_pred = tf.concat([new_pred_fake, new_pred_real], axis=0)
            drift_loss = self.loss_weights["drift"] * tf.reduce_mean(all_pred ** 2)

            d_loss = loss_fake + loss_real + gradient_penalty + drift_loss

            gradients = total_tape.gradient(d_loss, self.discriminator.trainable_weights)
            self.d_optimizer.apply_gradients(zip(gradients, self.discriminator.trainable_weights))

        if self.epoch == 30:
            self.generator.save("GEN_30.npy")
            self.discriminator.save("DIS_30.npy")

        # Update epoch
        self.epoch += 1

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
        elif weights_name in ['MNIST']:
            self.resolution = 32

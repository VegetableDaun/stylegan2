import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow import keras

from config_GAN import path_to_result, path_to_discriminator, path_to_generator, latent_dim, num_classes


class CustomCallback(keras.callbacks.Callback):
    def __init__(self, path=path_to_result, num_save=5, save_last=True, gen_images=True):
        super(CustomCallback, self).__init__()

        self.opt_cfg = {"learning_rate": 1e-3, "beta_1": 0.0, "beta_2": 0.99, "epsilon": 1e-8}

        self.num_save = num_save
        self.path = Path(path)
        self.save_last = save_last
        self.gen_images = gen_images

        try:
            for i in range(num_classes):
                os.makedirs(self.path / f'Image/class_{i}')
            os.makedirs(self.path / path_to_discriminator)
            os.makedirs(self.path / path_to_generator)
            os.makedirs(self.path / 'opt' / path_to_discriminator)
            os.makedirs(self.path / 'opt' / path_to_generator)
        except:
            pass

        if not os.path.isfile(self.path / 'data.json'):
            self.counter = 0
        else:
            with open(self.path / 'data.json', 'r') as F:
                self.counter = int(json.load(F))
                self.counter -= self.counter % self.num_save

        rnd = np.random.RandomState(666)
        self.noise = rnd.normal(size=(num_classes, latent_dim))
        self.labels = keras.utils.to_categorical(range(num_classes), num_classes)

    def on_train_begin(self, logs=None):
        self.model.counter = self.counter

        if os.path.isfile(self.path / 'metrics.json') and self.counter == 0:
            os.remove(self.path / 'metrics.json')
        elif self.counter != 0:
            self.model.load_opt_weights(self.path / 'opt' / path_to_discriminator / f'd_opt_{self.counter}.npy',
                                        self.path / 'opt' / path_to_generator / f'g_opt_{self.counter}.npy')

            if self.save_last:
                self.model.discriminator.load(self.path / 'Discriminator.npy')
                self.model.generator.load(self.path / 'Generator.npy')
            else:
                self.model.discriminator.load(self.path / path_to_discriminator / f'Discriminator_{self.counter}.npy')
                self.model.generator.load(self.path / path_to_generator / f'Generator_{self.counter}.npy')

    def on_epoch_begin(self, epoch, logs=None):
        if self.model.counter == self.model.T_e:
            self.model.g_optimizer.learning_rate = self.model.g_optimizer.learning_rate * 10
            # var.assign(var * value)
            self.model.d_optimizer.learning_rate = self.model.d_optimizer.learning_rate * 10

            # print(self.model.counter, self.counter, self.model.T_e)

    #
    #     if self.model.counter == self.model.T_e:
    #         new_STYLEGAN2 = StyleGan2(resolution=self.model.resolution, impl='cuda', gpu=True)
    #         # self.model.g_optimizer = tf.keras.optimizers.legacy.Adam(**self.opt_cfg)
    #         # self.model.d_optimizer = tf.keras.optimizers.legacy.Adam(**self.opt_cfg)
    #
    #         new_STYLEGAN2.generator = self.model.generator
    #         new_STYLEGAN2.discriminator = self.model.discriminator
    #
    #         new_STYLEGAN2.compile(
    #             d_optimizer=keras.optimizers.Adam(**self.opt_cfg),
    #             g_optimizer=keras.optimizers.Adam(**self.opt_cfg),
    #             T_s=self.model.T_s,
    #             T_e=self.model.T_e,
    #             counter=self.counter
    #         )
    #
    #         self.model = new_STYLEGAN2

    def on_epoch_end(self, epoch, logs=None):
        self.counter += 1
        self.model.counter += 1

        if not os.path.isfile(self.path / 'metrics.json'):
            with open(self.path / 'metrics.json', mode='w') as F:
                json.dump({self.counter: logs}, F)
        else:
            with open(self.path / 'metrics.json', mode='r') as F:
                feeds = dict(json.load(F))
                feeds[str(self.counter)] = logs
            with open(self.path / 'metrics.json', mode='w') as F:
                json.dump(feeds, F)

        self.model.save_opt_weights(self.path / 'opt' / path_to_discriminator / f'd_opt_{self.counter}',
                                    self.path / 'opt' / path_to_generator / f'g_opt_{self.counter}')

        if (epoch + 1) % self.num_save == 0:

            if self.save_last:
                self.model.discriminator.save(self.path / 'Discriminator.npy')
                self.model.generator.save(self.path / 'Generator.npy')
            else:
                self.model.discriminator.save(self.path / path_to_discriminator / f'Discriminator_{self.counter}.npy')
                self.model.generator.save(self.path / path_to_generator / f'Generator_{self.counter}.npy')

            if self.gen_images:
                img = self.model.generator(self.noise, c=self.labels)
                img = tf.transpose(img, [0, 2, 3, 1])
                for i in range(tf.shape(img)[0]):
                    img_i = np.array(img[i])
                    img_i = np.round(img_i * 255)
                    img_i = img_i.astype(np.uint8)
                    img_i = Image.fromarray(img_i)
                    img_i.save(self.path / f'Image/class_{i}/epoch_{self.counter}.jpg')

        with open(self.path / 'data.json', 'w') as F:
            json.dump(self.counter, F)

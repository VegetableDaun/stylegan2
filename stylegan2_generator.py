import numpy as np
import tensorflow as tf

from config_GAN import num_classes
from dnnlib.ops.upfirdn_2d import upsample_2d
from layers.dense_layer import DenseLayer
from layers.synthesis_main_layer import SynthesisMainLayer
from layers.to_rgb_layer import ToRgbLayer
from utils.utils_stylegan2 import nf
from utils.weights_map import available_weights, synthesis_weights, mapping_weights, weights_stylegan2_dir


class MappingNetwork(tf.keras.layers.Layer):
    """
    StyleGan2 generator mapping network, from z to dlatents for tensorflow 2.x
    """

    def __init__(self, resolution=1024, **kwargs):

        super(MappingNetwork, self).__init__(**kwargs)

        self.dlatent_size = 128
        self.dlatent_vector = (int(np.log2(resolution)) - 1) * 2
        self.mapping_layers = 8
        self.lrmul = 0.01

    def build(self, input_shape):

        self.weights_dict = {}
        for i in range(self.mapping_layers):
            setattr(self, 'Dense{}'.format(i),
                    DenseLayer(fmaps=128, lrmul=self.lrmul, name='Dense{}'.format(i)))

        self.Conditional_Dense = DenseLayer(fmaps=128, lrmul=self.lrmul, name='Conditional_Dense')

        self.g_mapping_broadcast = tf.keras.layers.RepeatVector(self.dlatent_vector)

    def call(self, z, lambda_t=1, c=None):

        assert c is not None, "Use conditional"

        z = tf.cast(z, 'float32')
        c = tf.cast(c, 'float32')

        # Normalize input z
        scale = tf.math.rsqrt(tf.reduce_mean(tf.square(z), axis=1, keepdims=True) + 1e-8)
        x = tf.math.multiply(z, scale)

        x = getattr(self, 'Dense{}'.format(0))(x)
        x = tf.math.multiply(tf.nn.leaky_relu(x, 0.2), tf.math.sqrt(2.))

        # Conditional
        x_c = self.Conditional_Dense(c)
        x_c = tf.math.multiply(tf.nn.leaky_relu(x_c, 0.2), tf.math.sqrt(2.))

        x = x + lambda_t * x_c

        # Mapping
        for i in range(1, self.mapping_layers):
            x = getattr(self, 'Dense{}'.format(i))(x)
            x = tf.math.multiply(tf.nn.leaky_relu(x, 0.2), tf.math.sqrt(2.))

        # Broadcasting
        dlatents = self.g_mapping_broadcast(x)

        return dlatents


class SynthesisNetwork(tf.keras.layers.Layer):
    """
    StyleGan2 generator synthesis network from dlatents to img tensor for tensorflow 2.x
    """

    def __init__(self, resolution=1024, impl='cuda', gpu=True, **kwargs):
        """
        Parameters
        ----------
        resolution : int, optional
            Resolution output of the synthesis network, will be parsed to the floor integer power of 2. 
            The default is 1024.
        impl : str, optional
            Wether to run some convolutions in custom tensorflow operations or cuda operations. 'ref' and 'cuda' available.
            The default is 'cuda'.
        gpu : boolean, optional
            Wether to use gpu. The default is True.

        """
        super(SynthesisNetwork, self).__init__(**kwargs)

        self.impl = impl
        self.gpu = gpu
        self.resolution = resolution

        self.resolution_log2 = int(np.log2(self.resolution))
        self.resample_kernel = [1, 3, 3, 1]

    def build(self, input_shape):

        # constant layer
        self.const_4_4 = self.add_weight(name='4x4/Const/const', shape=(1, 128, 4, 4),
                                         initializer=tf.random_normal_initializer(0, 1), trainable=True)
        # early layer 4x4
        self.layer_4_4 = SynthesisMainLayer(fmaps=nf(1), impl=self.impl, gpu=self.gpu, name='4x4')
        self.torgb_4_4 = ToRgbLayer(impl=self.impl, gpu=self.gpu, name='4x4')
        # main layers
        for res in range(3, self.resolution_log2 + 1):
            res_str = str(2 ** res)
            setattr(self, 'layer_{}_{}_up'.format(res_str, res_str),
                    SynthesisMainLayer(fmaps=nf(res - 1), impl=self.impl, gpu=self.gpu, up=True,
                                       name='{}x{}'.format(res_str, res_str)))
            setattr(self, 'layer_{}_{}'.format(res_str, res_str),
                    SynthesisMainLayer(fmaps=nf(res - 1), impl=self.impl, gpu=self.gpu,
                                       name='{}x{}'.format(res_str, res_str)))
            setattr(self, 'torgb_{}_{}'.format(res_str, res_str),
                    ToRgbLayer(impl=self.impl, gpu=self.gpu, name='{}x{}'.format(res_str, res_str)))

    def call(self, dlatents_in):

        dlatents_in = tf.cast(dlatents_in, 'float32')
        y = None

        # Early layers
        x = tf.tile(tf.cast(self.const_4_4, 'float32'), [tf.shape(dlatents_in)[0], 1, 1, 1])
        x = self.layer_4_4(x, dlatents_in[:, 0])
        y = self.torgb_4_4(x, dlatents_in[:, 1], y)

        # Main layers
        for res in range(3, self.resolution_log2 + 1):
            x = getattr(self, 'layer_{}_{}_up'.format(2 ** res, 2 ** res))(x, dlatents_in[:, res * 2 - 5])
            x = getattr(self, 'layer_{}_{}'.format(2 ** res, 2 ** res))(x, dlatents_in[:, res * 2 - 4])
            y = upsample_2d(y, k=self.resample_kernel, impl=self.impl, gpu=self.gpu)
            y = getattr(self, 'torgb_{}_{}'.format(2 ** res, 2 ** res))(x, dlatents_in[:, res * 2 - 3], y)

        y = tf.transpose(y, [0, 2, 3, 1])
        y = tf.keras.activations.sigmoid(y)
        images_out = tf.transpose(y, [0, 3, 1, 2])

        return tf.identity(images_out, name='images_out')


class StyleGan2Generator(tf.keras.layers.Layer):
    """
    StyleGan2 generator config f for tensorflow 2.x
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
            Wether to run some convolutions in custom tensorflow operations 
            or cuda operations. 'ref' and 'cuda' available.
            The default is 'cuda'.
        gpu : boolean, optional
            Wether to use gpu. The default is True.

        """
        super(StyleGan2Generator, self).__init__(**kwargs)

        self.resolution = resolution
        if weights is not None: self.__adjust_resolution(weights)

        self.mapping_network = MappingNetwork(resolution=self.resolution, name='Mapping_network')
        self.synthesis_network = SynthesisNetwork(resolution=self.resolution, impl=impl,
                                                  gpu=gpu, name='Synthesis_network')

        # load weights
        if weights is not None:
            # we run the network to define it, not the most efficient thing to do...
            _ = self(tf.zeros(shape=(1, 128)), lambda_t=1, c=tf.zeros(shape=(1, num_classes)))
            self.__load_weights(weights)

    def call(self, z, lambda_t=1, c=None):
        """

        Parameters
        ----------
        z : tensor, latent vector of shape [batch, 512]

        Returns
        -------
        img : tensor, image generated by the generator of shape  [batch, channel, height, width]

        """

        assert c is not None, "Use conditional"

        dlatents = self.mapping_network(z, lambda_t=lambda_t, c=c)
        img = self.synthesis_network(dlatents)

        return img

    def __adjust_resolution(self, weights_name):
        """
        Adjust resolution of the synthesis network output. 
        
        Parameters
        ----------
        weights_name : name of the weights

        Returns
        -------
        None.

        """
        if weights_name == 'ffhq':
            self.resolution = 1024
        elif weights_name == 'car':
            self.resolution = 512
        elif weights_name in ['cat', 'church', 'horse']:
            self.resolution = 256
        elif weights_name in ['MNIST']:
            self.resolution = 32

    def __load_weights(self, weights_name):
        """
        Load pretrained weights, stored as a dict with numpy arrays.
        Parameters
        ----------
        weights_name : name of the weights from best models

        Returns
        -------
        None.

        """

        if (weights_name in available_weights) and type(weights_name) == str:
            data = np.load(weights_stylegan2_dir + weights_name + '.npy', allow_pickle=True)[()]

            weights_mapping = [data.get(key) for key in mapping_weights]
            weights_synthesis = [data.get(key) for key in synthesis_weights[weights_name]]

            self.mapping_network.set_weights(weights_mapping)
            self.synthesis_network.set_weights(weights_synthesis)

            print("Loaded {} generator weights!".format(weights_name))
        else:
            raise Exception('Cannot load {} weights'.format(weights_name))

    def save(self, path_to_save):
        """
        Save pretrained weights as a dict with numpy arrays.
        Parameters
        ----------
        path_to_save : path where will be saved weight file with name and type of file

        """

        trainable_weights = self.mapping_network.trainable_weights + self.synthesis_network.trainable_weights
        data = {}
        for i in trainable_weights:
            data[i.name[i.name.find('/') + 1: len(i.name) - 2]] = i.numpy()

        # with open(path_to_save + '.npy', 'wb') as f:
        np.save(path_to_save, data, allow_pickle=True)

    def load(self, path_to_weights):
        """
        Load pretrained weights, stored as a dict with numpy arrays.
        Parameters
        ----------
        path_to_weights : path where store saved weight file with name and type of file

        Returns
        -------
        None.

        """

        if not self.mapping_network.trainable_weights or not self.synthesis_network.trainable_weights:
            _ = self(tf.zeros(shape=(1, 128)), lambda_t=1, c=tf.zeros(shape=(1, num_classes)))

        data = np.load(path_to_weights, allow_pickle=True)[()]

        weights_mapping = [data.get(key) for key in mapping_weights]
        weights_synthesis = [data.get(key) for key in synthesis_weights[self.resolution]]

        self.mapping_network.set_weights(weights_mapping)
        self.synthesis_network.set_weights(weights_synthesis)

        print("Loaded {} pixels discriminator weights!".format(self.resolution))

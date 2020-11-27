import tensorflow as tf
layers = tf.keras.layers
backend = tf.keras.backend
from tensorflow.keras.regularizers import l2
from tools import layers as custom_layers
from tools.Block_tricks import apRelu, global_context_block

class DenseNet(object):
    def __init__(self, params, growth_rate, reduction):
        assert len(params) == 4
        self.params = params
        self.reduction = reduction
        self.growth_rate =growth_rate

    def _conv_bn_relu(self, x, filters, kernel_size, strides=1):
        x = layers.Conv2D(filters, kernel_size, strides=strides, use_bias=False, kernel_initializer='he_normal', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = apRelu(x)
        return x

    def _bottleneck_block(self, x, growth_rate, dilation, name):

        x1 = layers.BatchNormalization(name=name + '_1_bn')(x)
        x1 = layers.Activation('swish')(x1)
        x1 = layers.Conv2D(2 * growth_rate, 1, use_bias=False, kernel_initializer='he_normal', name=name + '_1_conv')(x1)
        x1 = layers.BatchNormalization(name=name + '_2_bn')(x1)
        x1 = apRelu(x1)
        x1 = layers.Conv2D(growth_rate, 3, padding='same', use_bias=False, kernel_initializer='he_normal', dilation_rate=dilation, name=name + '_2_conv')(x1)
        x = layers.Concatenate(axis=-1, name=name + '_concat')([x, x1])
        return x

    def _dense_block(self, x, growth_rate, n_blocks, name, dilation=1):

        for i in range(n_blocks):
            x = self._bottleneck_block(x, growth_rate, dilation, name=name + '_block_' + str(i + 1))
        return x

    def _transition_block(self, x, reduction, name, dilation=1):

        x = layers.BatchNormalization(name=name + '_bn')(x)
        x = layers.Activation('swish')(x)
        x = layers.Conv2D(int(backend.int_shape(x)[-1] * reduction), 1, use_bias=False,  kernel_initializer='he_normal', name=name + '_conv')(x)

        if dilation == 1:
            x = layers.AveragePooling2D(2, strides=2, name=name + '_pool')(x)
        return x


    def __call__(self, inputs, output_stages='c5'):
        """
        call for DenseNet.
        :param inputs: a 4-D tensor.
        :param output_stages: str or a list of str containing the output stages.
        :return: the output of different stages.
        """
        x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)))(inputs)
        x = layers.Conv2D(64, 7, strides=2, use_bias=False, name='conv1/conv')(x)
        x = layers.BatchNormalization(
            axis=-1, epsilon=1.001e-5, name='conv1/bn')(x)
        x = layers.Activation('swish', name='conv1/swish')(x)
        x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
        x = layers.MaxPooling2D(3, strides=2, name='pool1')(x)

        c1 = x  # 64
        x = self._dense_block(x, self.growth_rate, self.params[0], name='D1')
        x = self._transition_block(x, self.reduction, name='T1')
        c2 = x # 32

        x = self._dense_block(x, self.growth_rate, self.params[1], name='D2')
        x = self._transition_block(x, self.reduction, name='T2')
        c3 = x  # 16

        x = self._dense_block(x, self.growth_rate, self.params[2], name='D3')
        x = self._transition_block(x, self.reduction, name='T3', dilation=2)
        c4 = x  # 16

        x = self._dense_block(x, self.growth_rate, self.params[3], name='D4', dilation=2)
        x = layers.BatchNormalization(name='bn')(x)
        x = apRelu(x)
        c5 = x # 16

        self.outputs = {'c1': c1, 'c2': c2, 'c3': c3,  'c4': c4, 'c5': c5}

        if type(output_stages) is not list:
            return self.outputs[output_stages]
        else:
            return [self.outputs[ci] for ci in output_stages]



class DeepLabV3Plus():
    def __init__(self, num_classes, encoder):

        dilation = [1, 2]

        self.dilation = dilation
        self.num_classes = num_classes
        self.encoder = encoder

    def __call__(self, inputs):

        return self._deeplab_v3_plus(inputs)

    def _deeplab_v3_plus(self, inputs):

        _, h, w, _ = backend.int_shape(inputs)
        self.aspp_size = (16, 16)

        c2, c5 = self.encoder(inputs, output_stages=['c1', 'c5'])
        # 64 ; 16

        x = self._aspp(c5, 256)
        x = layers.Dropout(rate=0.2)(x)

        x = layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(x)
        x = self._conv_bn_relu(x, 64, 1, strides=1)  # 64

        c2 = self._conv_bn_relu(c2, 64, 1, strides=1) # 64

        x = layers.concatenate([x, c2])
        x = self._conv_bn_relu(x, 256, 3, 1)
        x = layers.Dropout(rate=0.5)(x)

        x = self._conv_bn_relu(x, 256, 3, 1)
        x = layers.Dropout(rate=0.1)(x)

        x = layers.Conv2D(self.num_classes, 1, strides=1)(x)
        x = layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(x) # 256

        outputs = layers.Activation('softmax', name='softmax_out')(x)
        return  outputs

    def _conv_bn_relu(self, x, filters, kernel_size, strides=1):
        x = layers.Conv2D(filters, kernel_size, strides=strides, use_bias=False, kernel_initializer='he_normal', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = apRelu(x)
        return x

    def _aspp(self, x, out_filters):
        xs = list()
        x1 = layers.Conv2D(out_filters, 1, strides=1, use_bias=False, kernel_initializer='he_normal')(x)
        x1 = global_context_block(x1, out_filters // 4)
        xs.append(x1)

        for i in range(3):
            xi = layers.Conv2D(out_filters, 3, strides=1, padding='same',kernel_initializer='he_normal', dilation_rate=6 * (i + 1))(x)
            xi = global_context_block(xi, out_filters//4)
            xs.append(xi)
        img_pool = custom_layers.GlobalAveragePooling2D(keep_dims=True)(x)
        img_pool = layers.Conv2D(out_filters, 1, 1, kernel_initializer='he_normal')(img_pool)
        img_pool = layers.UpSampling2D(size=self.aspp_size, interpolation='bilinear')(img_pool)
        xs.append(img_pool)

        x = layers.Concatenate()(xs)
        x = layers.Conv2D(out_filters, 1, strides=1, kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        x = apRelu(x)

        return x


def Net(inputs, params, growth_rate, reduction, num_classes):
    encoder = DenseNet(params,growth_rate,reduction)
    v3P = DeepLabV3Plus(num_classes, encoder)
    output = v3P(inputs)
    return output



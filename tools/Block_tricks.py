import tensorflow as tf
layers = tf.keras.layers
backend = tf.keras.backend
from tensorflow.keras.regularizers import l2
from tools import layers as custom_layers


# An adaptively parametric rectifier linear unit (APReLU)
def apRelu(inputs):
    # get the number of channels
    channels = inputs.get_shape().as_list()[-1]
    # get a zero feature map
    zeros_input = layers.subtract([inputs, inputs])
    # get a feature map with only positive features
    pos_input = layers.Activation('relu')(inputs)
    # get a feature map with only negative features
    neg_input = layers.Minimum()([inputs,zeros_input])
    # define a network to obtain the scaling coefficients
    scales_p = layers.GlobalAveragePooling2D()(pos_input)
    scales_n = layers.GlobalAveragePooling2D()(neg_input)
    scales = layers.Concatenate()([scales_n, scales_p])
    scales = layers.Dense(channels//4, activation='linear', kernel_initializer='he_normal')(scales)
    scales = layers.BatchNormalization(momentum=0.9)(scales)
    scales = layers.Activation('relu')(scales)
    scales = layers.Dense(channels, activation='linear', kernel_initializer='he_normal')(scales)
    scales = layers.BatchNormalization(momentum=0.9)(scales)
    scales = layers.Activation('sigmoid')(scales)
    scales = layers.Reshape((1,1,channels))(scales)
    # apply a paramtetric relu
    neg_part = layers.multiply([scales, neg_input])
    return layers.add([pos_input, neg_part])

def _conv_bn_relu(x, filters, kernel_size, strides=1):
    x = layers.Conv2D(filters, kernel_size, strides=strides, use_bias=False, kernel_initializer='he_normal',
                       padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = apRelu(x)
    return x

def backend_expand_dims_1(x):
    return backend.expand_dims(x, axis=1)

def backend_expand_dims_last( x):
    return backend.expand_dims(x, axis=-1)

def backend_dot(x):
    return backend.batch_dot(x[0], x[1])

def global_context_block(x, channels):
    bs, h, w, c = x.shape.as_list()
    input_x = x
    input_x = layers.Reshape((-1, c))(input_x)  # [N, H*W, C]
    input_x = layers.Permute((2, 1))(input_x)  # [N, C, H*W]
    input_x = layers.Lambda(backend_expand_dims_1)(input_x)  # [N, 1, C, H*W]

    context_mask = layers.Conv2D(1, (1, 1))(x)
    context_mask = layers.Reshape((-1, 1))(context_mask)  # [N, H*W, 1]
    context_mask = layers.Softmax(axis=1)(context_mask)  # [N, H*W, 1]
    context_mask = layers.Permute((2, 1))(context_mask)  # [N, 1, H*W]
    context_mask = layers.Lambda(backend_expand_dims_last)(context_mask)  # [N, 1, H*W, 1]

    context = layers.Lambda(backend_dot)([input_x, context_mask])
    context = layers.Reshape((1, 1, c))(context)  # [N, 1, 1, C]

    context_transform = _conv_bn_relu(context, channels, 1, strides=1)
    context_transform = layers.Conv2D(c, (1, 1))(context_transform)
    context_transform = layers.Activation('sigmoid')(context_transform)
    x = layers.Multiply()([x, context_transform])

    context_transform = _conv_bn_relu(context, channels, 1, strides=1)
    context_transform = layers.Conv2D(c, (1, 1))(context_transform)
    x = layers.add([x, context_transform])

    return x
    # -----
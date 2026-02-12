import os
os.environ["KERAS_BACKEND"] = "jax"

import numpy as np
import keras

from src.model.sub_layer import conv_1d_1x1, conv_1d_1x3, conv_1d_1x5, conv_1d_1x7, max_pool_1d_to_1x1
from src.model.sub_layer import conv_2d_1x1, conv_2d_1x3, conv_2d_1x5, max_pool_2d_to_1x1


@keras.saving.register_keras_serializable()
def gelu_approximate(x):
    return keras.ops.gelu(x, approximate=True)

class PositionalEncoding(keras.layers.Layer):
    def __init__(self, position, d_model, **kwargs):
        """
        포지셔널 인코딩 레이어를 초기화합니다.

        Args:
            position (int): 시퀀스의 최대 길이 (최대 문장 길이)
            d_model (int): 임베딩 벡터의 차원
        """
        super(PositionalEncoding, self).__init__(**kwargs) # **kwargs 전달
        self.position = position
        self.d_model = d_model
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        """
        각도 계산을 위한 내부 함수
        """
        angles = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return position * angles

    def positional_encoding(self, position, d_model):
        """
        포지셔널 인코딩 행렬을 생성합니다.
        """
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis],
                                     np.arange(d_model)[np.newaxis, :],
                                     d_model)

        # 짝수 인덱스에는 사인 함수 적용
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        # 홀수 인덱스에는 코사인 함수 적용
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, inputs):
        """
        레이어의 정방향 계산을 수행합니다.
        입력 텐서에 포지셔널 인코딩을 더합니다.
        """
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

    def get_config(self):
        config = super(PositionalEncoding, self).get_config()
        config.update({'position': self.position,
                       'd_model': self.d_model})

        return config


class InceptionBlock1D(keras.layers.Layer):
    def __init__(self, output_dim_1x1=64, hidden_dim_3x3=96, output_dim_3x3=128, hidden_dim_5x5=16, output_dim_5x5=32,
                 hidden_dim_7x7=24, output_dim_7x7=32, output_dim_max_pool=32, dropout_rate=0.2, **kwargs):
        super(InceptionBlock1D, self).__init__(**kwargs)
        self.output_dim_1x1 = output_dim_1x1
        self.hidden_dim_3x3 = hidden_dim_3x3
        self.output_dim_3x3 = output_dim_3x3
        self.hidden_dim_5x5 = hidden_dim_5x5
        self.output_dim_5x5 = output_dim_5x5
        self.hidden_dim_7x7 = hidden_dim_7x7
        self.output_dim_7x7 = output_dim_7x7
        self.output_dim_max_pool = output_dim_max_pool
        self.dropout_rate = dropout_rate

        self.conv_1x1 = conv_1d_1x1(output_dim=self.output_dim_1x1, dropout_rate=self.dropout_rate)
        self.conv_3x3 = conv_1d_1x3(hidden_dim=self.hidden_dim_3x3, output_dim=self.output_dim_3x3, dropout_rate=self.dropout_rate)
        self.conv_5x5 = conv_1d_1x5(hidden_dim=self.hidden_dim_5x5, output_dim=self.output_dim_5x5, dropout_rate=self.dropout_rate)
        self.conv_7x7 = conv_1d_1x7(hidden_dim=self.hidden_dim_7x7, output_dim=self.output_dim_7x7)
        self.max_pool = max_pool_1d_to_1x1(output_dim=self.output_dim_max_pool, dropout_rate=self.dropout_rate)

    def call(self, inputs_layer):
        output_layer_1 = self.conv_1x1(inputs_layer)
        output_layer_2 = self.conv_3x3(inputs_layer)
        output_layer_3 = self.conv_5x5(inputs_layer)
        output_layer_4 = self.conv_7x7(inputs_layer)
        output_layer_5 = self.max_pool(inputs_layer)

        return keras.layers.concatenate(inputs=[output_layer_1, output_layer_2, output_layer_3, output_layer_4, output_layer_5], axis=2)

    def get_config(self):
        config = super().get_config()
        config.update({'output_dim_1x1': self.output_dim_1x1, 'hidden_dim_3x3': self.hidden_dim_3x3, 'output_dim_3x3': self.output_dim_3x3,
                       'hidden_dim_5x5': self.hidden_dim_5x5, 'output_dim_5x5': self.output_dim_5x5,
                       'output_dim_max_pool': self.output_dim_max_pool, 'dropout_rate': self.dropout_rate})

        return config


class TransformerEncoderBlock(keras.layers.Layer):
    def __init__(self, head_size, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.head_size = head_size
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate

        self.attention = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=head_size)
        self.dropout1 = keras.layers.Dropout(dropout_rate)
        self.norm1 = keras.layers.LayerNormalization(epsilon=1e-6)

        self.ffn_dense1 = keras.layers.Dense(ff_dim, activation="gelu")
        self.ffn_dense2 = keras.layers.Dense(head_size, activation="linear")
        self.dropout2 = keras.layers.Dropout(dropout_rate)
        self.norm2 = keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        attention_output = self.attention(query=inputs, value=inputs, key=inputs)
        attention_output = self.dropout1(attention_output)
        out1 = inputs + attention_output
        norm_out1 = self.norm1(out1)

        ffn_output = self.ffn_dense1(norm_out1)
        ffn_output = self.ffn_dense2(ffn_output)
        ffn_output = self.dropout2(ffn_output)
        ffn_output = norm_out1 + ffn_output
        ffn_output = self.norm2(ffn_output)

        return ffn_output

    def get_config(self):
        config = super().get_config()
        config.update({'head_size': self.head_size,
                       'num_heads': self.num_heads,
                       'ff_dim': self.ff_dim,
                       'dropout_rate': self.dropout_rate})

        return config

class InceptionBlock2D(keras.layers.Layer):
    def __init__(self, output_dim_1x1=64, hidden_dim_3x3=96, output_dim_3x3=128, hidden_dim_5x5=16, output_dim_5x5=32, output_dim_max_pool=32,
                 dropout_rate=0.2, **kwargs):
        super(InceptionBlock2D, self).__init__(**kwargs)
        self.output_dim_1x1 = output_dim_1x1
        self.hidden_dim_3x3 = hidden_dim_3x3
        self.output_dim_3x3 = output_dim_3x3
        self.hidden_dim_5x5 = hidden_dim_5x5
        self.output_dim_5x5 = output_dim_5x5
        self.output_dim_max_pool = output_dim_max_pool
        self.dropout_rate = dropout_rate

        self.conv_1x1 = conv_2d_1x1(output_dim=self.output_dim_1x1, dropout_rate=self.dropout_rate)
        self.conv_3x3 = conv_2d_1x3(hidden_dim=self.hidden_dim_3x3, output_dim=self.output_dim_3x3, dropout_rate=self.dropout_rate)
        self.conv_5x5 = conv_2d_1x5(hidden_dim=self.hidden_dim_5x5, output_dim=self.output_dim_5x5, dropout_rate=self.dropout_rate)
        self.max_pool = max_pool_2d_to_1x1(output_dim=self.output_dim_max_pool, dropout_rate=self.dropout_rate)

    def call(self, inputs_layer):

        output_layer_1 = self.conv_1x1(inputs_layer)
        output_layer_2 = self.conv_3x3(inputs_layer)
        output_layer_3 = self.conv_5x5(inputs_layer)
        output_layer_4 = self.max_pool(inputs_layer)

        result = keras.layers.concatenate(inputs=[output_layer_1, output_layer_2, output_layer_3, output_layer_4], axis=1)

        return result

    def get_config(self):
        config = super().get_config()
        config.update({'output_dim_1x1': self.output_dim_1x1, 'hidden_dim_3x3': self.hidden_dim_3x3, 'output_dim_3x3': self.output_dim_3x3,
                       'hidden_dim_5x5': self.hidden_dim_5x5, 'output_dim_5x5': self.output_dim_5x5, 'output_dim_max_pool': self.output_dim_max_pool,
                       'dropout_rate': self.dropout_rate})

        return config


@keras.saving.register_keras_serializable()
class DecompositionLayer(keras.layers.Layer):
    """
    이동 평균을 사용하여 시계열을 추세와 계절성 성분으로 분해합니다.
    """
    def __init__(self, kernel_size, **kwargs):
        super(DecompositionLayer, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.avg = keras.layers.AvgPool1D(pool_size=kernel_size, strides=1, padding='same')

    def call(self, x):
        trend = self.avg(x)
        seasonal = x - trend
        return seasonal, trend


@keras.saving.register_keras_serializable()
class FeatureWiseScalingLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.activation = keras.layers.Activation('gelu')
        self.scaling_vector = None

    def build(self, input_shape):
        feature_dim = input_shape[-1]
        self.scaling_vector = self.add_weight(shape=(feature_dim,), initializer='ones', trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        y = self.activation(self.scaling_vector)
        y = inputs * y

        return y

    def compute_output_shape(self, input_shape):
        return input_shape
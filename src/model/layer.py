import os
os.environ["KERAS_BACKEND"] = "jax"

import numpy as np
import keras

from src.model.sub_layer import conv_1d_1x1, conv_1d_1x3, conv_1d_1x5, conv_1d_1x7, max_pool_1d_to_1x1
from src.model.sub_layer import conv_2d_1x1, conv_2d_1x3, conv_2d_1x5, max_pool_2d_to_1x1


@keras.saving.register_keras_serializable()
def gelu_approximate(x):
    return keras.ops.gelu(x, approximate=True)

@keras.saving.register_keras_serializable()
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
        return keras.ops.cast(pos_encoding, dtype='float32')

    def call(self, inputs):
        """
        레이어의 정방향 계산을 수행합니다.
        입력 텐서에 포지셔널 인코딩을 더합니다.
        """
        return inputs + self.pos_encoding[:, :keras.ops.shape(inputs)[1], :]

    def get_config(self):
        config = super(PositionalEncoding, self).get_config()
        config.update({'position': self.position,
                       'd_model': self.d_model})

        return config


@keras.saving.register_keras_serializable()
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
        self.conv_7x7 = conv_1d_1x7(hidden_dim=self.hidden_dim_7x7, output_dim=self.output_dim_7x7, dropout_rate=self.dropout_rate)
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
                       'hidden_dim_7x7': self.hidden_dim_7x7, 'output_dim_7x7': self.output_dim_7x7,
                       'output_dim_max_pool': self.output_dim_max_pool, 'dropout_rate': self.dropout_rate})

        return config


@keras.saving.register_keras_serializable()
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
        # ffn_dense2 는 잔차 연결을 위해 입력과 같은 차원으로 되돌려야 하므로 build() 에서 생성한다.
        self.ffn_dense2 = None
        self.dropout2 = keras.layers.Dropout(dropout_rate)
        self.norm2 = keras.layers.LayerNormalization(epsilon=1e-6)

    def build(self, input_shape):
        # head_size 는 어텐션 헤드 하나의 key 차원일 뿐 모델 차원이 아니다.
        # FFN 출력은 입력 feature 차원과 같아야 norm_out1 + ffn_output 잔차 덧셈이 성립한다.
        self.ffn_dense2 = keras.layers.Dense(input_shape[-1], activation="linear")
        super().build(input_shape)

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

@keras.saving.register_keras_serializable()
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
class MultiScaleMean(keras.layers.Layer):
    """
    입력 창에서 여러 길이의 이동평균을 하나씩 뽑아 벡터로 만듭니다.

    스케일 s 에 대해 창의 "마지막 s 스텝" 평균을 구합니다. 스케일이 커질수록
    더 오래된 구간까지 포함하므로, 짧은 평균과 긴 평균의 차이가 곧 추세 정보가 됩니다.
    s 가 창 길이와 같으면 창 전체 평균입니다.

    Args:
        scales: 평균을 낼 길이들. 각 값은 1 이상 seq_len 이하여야 합니다.

    입력 (batch, seq_len, n_features) → 출력 (batch, len(scales) * n_features)
    """
    def __init__(self, scales=(3, 5, 10, 15, 20, 25, 30), **kwargs):
        super().__init__(**kwargs)
        self.scales = tuple(int(s) for s in scales)

        if len(self.scales) == 0:
            raise ValueError('scales 는 비어 있을 수 없습니다.')
        if any(s < 1 for s in self.scales):
            raise ValueError(f'scales 의 모든 값은 1 이상이어야 합니다. (받은 값: {self.scales})')

    def build(self, input_shape):
        seq_len = input_shape[1]

        if seq_len is not None and max(self.scales) > seq_len:
            raise ValueError(f'scales 의 값은 입력 창 길이({seq_len}) 이하여야 합니다. '
                             f'(받은 값: {self.scales})')
        super().build(input_shape)

    def call(self, inputs):
        means = [keras.ops.mean(inputs[:, -s:, :], axis=1) for s in self.scales]

        return keras.ops.concatenate(means, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], len(self.scales) * input_shape[-1])

    def get_config(self):
        config = super().get_config()
        config.update({'scales': list(self.scales)})

        return config


@keras.saving.register_keras_serializable()
class ScaleWiseAffine(keras.layers.Layer):
    """
    입력 원소마다 독립적인 a*x + b 를 적용합니다.

    MultiScaleMean 뒤에 붙이면 스케일마다 기울기 a 와 절편 b 를 따로 학습합니다.
    학습된 a, b 를 그대로 읽어 각 스케일이 예측에 얼마나 기여하는지 볼 수 있습니다.

    a 의 초기값은 1/n 입니다. 이렇게 두면 학습 시작 시점의 출력이 여러 평균의 평균,
    즉 "현재 수준을 그대로 예측"이 되어 물리적으로 타당한 출발점이 됩니다.
    a 를 1 로 두면 초기 출력이 스케일 개수배로 커져 학습이 크게 흔들립니다.

    Args:
        l2 (float): a 에 걸 L2 벌점. 0 이면 걸지 않습니다(기본값).
            중첩된 이동평균은 서로 상관이 매우 높아(이 데이터에서 최소 0.9947,
            조건수 4139) 벌점 없이 학습하면 계수가 불안정해집니다.
    """
    def __init__(self, l2=0.0, **kwargs):
        super().__init__(**kwargs)
        self.l2 = float(l2)
        self.a = None
        self.b = None

    def build(self, input_shape):
        n_units = input_shape[-1]
        regularizer = keras.regularizers.L2(self.l2) if self.l2 > 0 else None
        self.a = self.add_weight(name='a', shape=(n_units,),
                                 initializer=keras.initializers.Constant(1.0 / n_units),
                                 regularizer=regularizer, trainable=True)
        self.b = self.add_weight(name='b', shape=(n_units,), initializer='zeros', trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        return inputs * self.a + self.b

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({'l2': self.l2})

        return config


@keras.saving.register_keras_serializable()
class FeatureWiseScalingLayer(keras.layers.Layer):
    """
    특징(feature)마다 학습되는 스케일 값을 곱해 주는 레이어입니다.

    스케일 벡터를 그대로 곱합니다. 이전에는 gelu 를 거친 값을 곱했는데,
    gelu 의 최솟값이 -0.17 이라 스케일이 그 아래로 내려갈 수 없었습니다.
    즉 특징의 부호를 뒤집거나 크게 음수로 만드는 것이 불가능했고,
    스케일 값이 음수로 많이 밀리면 gelu 의 기울기가 0 에 가까워져
    한 번 죽은 특징이 되살아나지 못했습니다.

    이제 스케일은 실수 전체 범위를 자유롭게 쓸 수 있고,
    initializer='ones' 이므로 학습 시작 시점의 스케일은 정확히 1.0 입니다.
    (gelu 를 거치던 때에는 gelu(1)=0.841 이 초기 스케일이었습니다.)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.scaling_vector = None

    def build(self, input_shape):
        feature_dim = input_shape[-1]
        self.scaling_vector = self.add_weight(shape=(feature_dim,), initializer='ones', trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        return inputs * self.scaling_vector

    def compute_output_shape(self, input_shape):
        return input_shape
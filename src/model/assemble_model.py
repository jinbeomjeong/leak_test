import os
os.environ["KERAS_BACKEND"] = "jax"

import keras

from src.model.layer import gelu_approximate, FeatureWiseScalingLayer
from src.model.model import time_mixer_block


def build_reg_model(input_shape, d_dims=64, dropout_rate=0.2, learning_rate=0.001):
    input_layer = keras.layers.Input(shape=input_shape)

    x_res = keras.layers.Dense(units=d_dims, activation='linear')(input_layer)

    for i in range(3):
        # conv → norm → activation 순서.
        # 기존에는 Conv1D(activation=...) 로 활성화를 먼저 걸고 LayerNormalization 을 뒤에 두어
        # 정규화가 활성화 결과를 다시 중심 이동시키고 있었다.
        x = keras.layers.Conv1D(filters=d_dims, kernel_size=3, padding='causal')(x_res)
        x = keras.layers.LayerNormalization()(x)
        x = keras.layers.Activation(gelu_approximate)(x)
        x = keras.layers.Dropout(dropout_rate)(x)

        # 두 번째 conv 의 활성화는 잔차 덧셈 뒤에 온다 (ResNet 의 conv→norm→add→activation 형태).
        x = keras.layers.Conv1D(filters=d_dims, kernel_size=3, padding='causal')(x)
        x = keras.layers.LayerNormalization()(x)
        x_res = keras.layers.Activation(gelu_approximate)(x+x_res)

    y = keras.layers.Flatten()(x_res)
    y = keras.layers.Dropout(dropout_rate)(y)

    y = FeatureWiseScalingLayer()(y)
    y = keras.layers.Dense(units=input_shape[0]*3, activation='linear')(y)

    y_res = time_mixer_block(input_layer=y, pred_len=input_shape[0]*3, dropout_rate=dropout_rate)
    y_res = keras.layers.LayerNormalization()(y_res)
    y = y+y_res

    y = FeatureWiseScalingLayer()(y)

    y = keras.layers.Dense(units=1, activation='linear')(y)

    model = keras.models.Model(inputs=input_layer, outputs=y)

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss=keras.losses.LogCosh(),
                  metrics=['mean_absolute_error', 'mean_absolute_percentage_error'])

    return model
import os
os.environ["KERAS_BACKEND"] = "jax"

import keras
#from sub_layer import fft_for_period
from src.model.layer import DecompositionLayer, gelu_approximate
from src.model.sub_layer import count_divisions_by_two


# class TimesNetBlock(keras.layers.Layer):
#     def __init__(self, seq_len=1, pred_len=1,  top_k=3, dropout_rate=0.1, **kwargs):
#         super(TimesNetBlock, self).__init__(**kwargs)
#         self.seq_len = seq_len
#         self.pred_len = pred_len
#         self.k = top_k
#         self.dropout_rate = dropout_rate
#
#     def build(self, input_shape):
#         inception_block_1 = InceptionBlock2D(output_dim_1x1=32, hidden_dim_3x3=16, output_dim_3x3=32, hidden_dim_5x5=16, output_dim_5x5=32,
#                                              output_dim_max_pool=32, dropout_rate=self.dropout_rate)
#
#         inception_block_2 = InceptionBlock2D(output_dim_1x1=64, hidden_dim_3x3=32, output_dim_3x3=64, hidden_dim_5x5=32,
#                                              output_dim_5x5=64, output_dim_max_pool=64, dropout_rate=self.dropout_rate)
#
#         self.conv = tf.keras.Sequential([inception_block_1,
#                                          keras.layers.MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding='same'),
#                                          inception_block_2,
#                                          keras.layers.Activation('gelu')])
#
#     def call(self, x):
#         B, T, N = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
#         period_list, period_weight = fft_for_period(x, self.k)
#
#         res = []
#
#         for i in range(self.k):
#             period = period_list[i]
#             total_len = self.seq_len + self.pred_len
#             out = x
#
#             if total_len % period != 0:
#                 length = tf.cast(((total_len // period) + 1) * period, dtype=tf.int32)
#                 padding = tf.zeros([B, length - total_len, N])
#                 out = tf.concat([x, padding], axis=1)
#             else:
#                 length = total_len
#                 # out = x
#
#             # Reshape
#             out = tf.reshape(tensor=out, shape=[B, length // period, period, N])
#             out = tf.transpose(a=out, perm=[0, 3, 1, 2]) # [B, N, num_periods, period_len]
#
#             # 2D Conv
#             out = self.conv(out)
#
#             # Reshape back
#             out = tf.transpose(a=out, perm=[0, 2, 3, 1]) # [B, num_periods, period_len, N]
#             out = tf.reshape(tensor=out, shape=[B, -1, N])
#
#             res.append(out[:, :total_len, :])
#
#         res = tf.stack(res, axis=-1) # [B, T, N, k]
#
#         # Adaptive Aggregation
#         period_weight = tf.nn.softmax(period_weight, axis=1)
#         period_weight = tf.expand_dims(tf.expand_dims(period_weight, axis=1), axis=1) # [B, 1, 1, k]
#         period_weight = tf.tile(period_weight, [1, T, N, 1]) # [B, T, N, k]
#
#         res = tf.reduce_sum(res * period_weight, axis=-1)
#
#         return res
#
#     def get_config(self):
#         config = super(TimesNetBlock, self).get_config()
#         config.update({'seq_len': self.seq_len,
#                        'pred_len': self.pred_len,
#                        'top_k': self.k,
#                        'dropout_rate': self.dropout_rate,})
#         return config


def time_mixer_block(input_layer, pred_len=1, go_backward=False, dropout_rate=0.2):
    input_raw = keras.ops.reverse(input_layer, axes=1) if go_backward else input_layer
    input_raw = keras.ops.expand_dims(input_raw, axis=2)

    multi_scale_input_list = [input_raw]

    for i in range(count_divisions_by_two(input_raw.shape[1])-1):
        i = (i*2)+2
        avg_layer = keras.layers.AveragePooling1D(pool_size=i, strides=i, padding='valid')(input_raw)
        multi_scale_input_list.append(avg_layer)

    seasonal_list = []
    trend_list = []

    for multi_scale_input_layer in multi_scale_input_list:
        seasonal, trend = DecompositionLayer(kernel_size=3)(multi_scale_input_layer)

        seasonal = keras.ops.squeeze(seasonal, axis=2)
        seasonal_output = keras.layers.Dense(units=multi_scale_input_layer.shape[1], activation='linear')(seasonal)
        seasonal_output = keras.layers.Dropout(dropout_rate)(seasonal_output)
        seasonal_list.append(seasonal_output)

        trend = keras.ops.squeeze(trend, axis=2)
        trend_output = keras.layers.Dense(units=multi_scale_input_layer.shape[1], activation='linear')(trend)
        trend_output = keras.layers.Dropout(dropout_rate)(trend_output)
        trend_list.append(trend_output)

        #output_list.append(keras.layers.Add()([seasonal_output, trend_output]))

    output_1 = seasonal_list[0]
    seasonal_mix_list = [output_1]

    for i in range(len(seasonal_list)-1):
        output_1 = keras.layers.Dense(units=seasonal_list[i+1].shape[1], activation='linear')(output_1)
        output_1 = keras.layers.LayerNormalization()(output_1)
        output_1 = keras.layers.Dropout(dropout_rate)(output_1)
        output_1 = keras.layers.Activation(gelu_approximate)(output_1) #gelu
        output_1 = keras.layers.add([output_1, seasonal_list[i+1]])
        seasonal_mix_list.append(output_1)

    trend_list.reverse()
    output_2 = trend_list[0]
    trend_mix_list = [output_2]

    for i in range(len(trend_list)-1):
        output_2 = keras.layers.Dense(units=trend_list[i+1].shape[1], activation='linear')(output_2)
        output_2 = keras.layers.LayerNormalization()(output_2)
        output_2 = keras.layers.Dropout(dropout_rate)(output_2)
        output_2 = keras.layers.Activation(gelu_approximate)(output_2) #gelu
        output_2 = keras.layers.add([output_2, trend_list[i+1]])
        trend_mix_list.append(output_2)

    trend_mix_list.reverse()

    mix_output_list = []
    hidden_units = 128

    for seasonal_mix_layer, trend_mix_layer in zip(seasonal_mix_list, trend_mix_list):
        mix_output = seasonal_mix_layer+trend_mix_layer
        # 비선형성은 은닉층에 둔다. 위쪽 seasonal/trend 믹싱 루프와 같은 Dense→LN→Dropout→Activation 순서.
        mix_output = keras.layers.Dense(units=hidden_units, activation='linear')(mix_output)
        mix_output = keras.layers.LayerNormalization()(mix_output)
        mix_output = keras.layers.Dropout(dropout_rate)(mix_output)
        mix_output = keras.layers.Activation(gelu_approximate)(mix_output)
        # 예측을 내보내는 투영층은 linear 여야 한다. gelu 를 쓰면 출력이 -0.17 아래로 내려갈 수 없어
        # 음수 쪽 누설 유량(타깃 범위 -7.5 ~ +7.2)을 표현하지 못한다.
        mix_output = keras.layers.Dense(units=pred_len, activation='linear')(mix_output)
        mix_output_list.append(mix_output)

    return keras.layers.add(mix_output_list)
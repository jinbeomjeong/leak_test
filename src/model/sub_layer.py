import os
os.environ["KERAS_BACKEND"] = "jax"

import keras


def count_linear_scales(seq_len, step=2, min_pooled_len=8):
    """
    time_mixer_block 이 만들 멀티스케일 다운샘플링 단계의 개수를 셉니다.

    pool_size 를 step, 2*step, 3*step ... 처럼 선형으로 키워가며,
    풀링 후 길이가 min_pooled_len 이상 남는 동안만 스케일을 추가합니다.

    기존에는 개수를 log2 기반(count_divisions_by_two)으로 세면서 실제 pool_size 는
    선형으로 키워 두 규칙이 서로 맞지 않았습니다. 여기서는 둘 다 선형으로 통일합니다.

    Args:
        seq_len (int): 원본 시퀀스 길이.
        step (int): pool_size 증가 폭. pool_size 는 step, 2*step, 3*step ... 순서.
        min_pooled_len (int): 풀링 후 남아야 하는 최소 길이.
            DecompositionLayer 의 이동평균(kernel_size=3)이 의미를 가지려면
            어느 정도 길이가 남아 있어야 하므로 하한을 둡니다.

    Returns:
        int: 추가할 스케일의 개수. 길이가 짧아 축소할 수 없으면 0.
    """
    count = 0
    pool_size = step

    while seq_len // pool_size >= min_pooled_len:
        count += 1
        pool_size += step

    return count


def transformer_decoder(inputs, encoder_outputs, head_size, num_heads, ff_dim, dropout=0):
    # 1. Masked Multi-Head Self-Attention
    # use_causal_mask=True 로 미래 시점의 정보 참조 방지
    self_attention = keras.layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs, use_causal_mask=True)
    attention1 = keras.layers.LayerNormalization(epsilon=1e-6)(inputs + self_attention)

    # 2. Encoder-Decoder Cross-Attention
    cross_attention = keras.layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(attention1, encoder_outputs, encoder_outputs) # Query: decoder, Key/Value: encoder
    attention2 = keras.layers.LayerNormalization(epsilon=1e-6)(attention1 + cross_attention)

    # 3. Feed Forward Network
    ffn = keras.Sequential([keras.layers.Dense(ff_dim, activation="gelu"),
                            keras.layers.Dense(inputs.shape[-1]),])
    x_ffn = ffn(attention2)
    x = keras.layers.Dropout(dropout)(x_ffn)
    decoder_output = keras.layers.LayerNormalization(epsilon=1e-6)(attention2 + x)

    return decoder_output


# def fft_for_period(x, k=2):
#     x_transposed = tf.transpose(x, perm=[0, 2, 1])  # tf.signal.rfft는 마지막 축에 대해 수행되므로, 축을 변경해야 함 [B, T, C] -> [B, C, T]
#     xf = tf.signal.rfft(x_transposed) # [B, C, F]
#
#     # 주파수 리스트 (진폭 기준)
#     frequency_list = tf.math.abs(xf)
#     frequency_list = tf.reduce_mean(frequency_list, axis=1) # [B, F]
#     frequency_list = tf.reduce_mean(frequency_list, axis=0) # [F]
#
#     # 첫 번째 주파수(DC 성분)는 무시
#     frequency_list = tf.tensor_scatter_nd_update(frequency_list, [[0]], [0.0])
#
#     _, top_list = tf.math.top_k(frequency_list, k=k)
#
#     # 주파수 인덱스를 주기로 변환
#     T = tf.cast(tf.shape(x)[1], dtype=tf.float32)
#     # 0으로 나누는 것을 방지하기 위해 작은 값(epsilon) 추가
#     period = T / (tf.cast(top_list, dtype=tf.float32) + 1e-8)
#     period = tf.cast(tf.math.round(period), dtype=tf.int32)
#
#     # top_k 주파수의 진폭(가중치) 계산
#     amplitudes = tf.math.abs(xf) # [B, C, F]
#     amplitudes = tf.reduce_mean(amplitudes, axis=1) # [B, F]
#     period_weight = tf.gather(amplitudes, top_list, axis=1) # [B, k]
#
#     return period, period_weight


def conv_1d_1x1(output_dim=8, dropout_rate=0.2):
    model = keras.Sequential([keras.layers.Conv1D(filters=output_dim, kernel_size=1, activation=None, padding='same'),
                              keras.layers.BatchNormalization(),
                              keras.layers.Activation('gelu'),
                              keras.layers.Dropout(dropout_rate)])
    return model

def conv_1d_1x3(hidden_dim=4, output_dim=8, dropout_rate=0.2):
    model = keras.Sequential([keras.layers.Conv1D(filters=hidden_dim, kernel_size=1, activation=None, padding='same'),
                              keras.layers.BatchNormalization(),
                              keras.layers.Activation('gelu'),
                              keras.layers.Dropout(dropout_rate),
                              keras.layers.Conv1D(filters=output_dim, kernel_size=3, activation=None, padding='same'),
                              keras.layers.BatchNormalization(),
                              keras.layers.Activation('gelu'),
                              keras.layers.Dropout(dropout_rate)])
    return model

def conv_1d_1x5(hidden_dim=4, output_dim=8, dropout_rate=0.2):
    model = keras.Sequential([keras.layers.Conv1D(filters=hidden_dim, kernel_size=1, activation=None, padding='same'),
                              keras.layers.BatchNormalization(),
                              keras.layers.Activation('gelu'),
                              keras.layers.Dropout(dropout_rate),
                              keras.layers.Conv1D(filters=output_dim, kernel_size=5, activation=None, padding='same'),
                              keras.layers.BatchNormalization(),
                              keras.layers.Activation('gelu'),
                              keras.layers.Dropout(dropout_rate)])
    return model

def conv_1d_1x7(hidden_dim=4, output_dim=8, dropout_rate=0.2):
    model = keras.Sequential([keras.layers.Conv1D(filters=hidden_dim, kernel_size=1, activation=None, padding='same'),
                              keras.layers.BatchNormalization(),
                              keras.layers.Activation('gelu'),
                              keras.layers.Dropout(dropout_rate),
                              keras.layers.Conv1D(filters=output_dim, kernel_size=7, activation=None, padding='same'),
                              keras.layers.BatchNormalization(),
                              keras.layers.Activation('gelu'),
                              keras.layers.Dropout(dropout_rate)])
    return model

def max_pool_1d_to_1x1(output_dim=8, dropout_rate=0.2):
    model = keras.Sequential([keras.layers.MaxPooling1D(pool_size=3, strides=1, padding='same'),
                              keras.layers.Conv1D(filters=output_dim, kernel_size=1, activation=None, padding='same'),
                              keras.layers.BatchNormalization(),
                              keras.layers.Activation('gelu'),
                              keras.layers.Dropout(dropout_rate)])
    return model

def conv_2d_1x1(output_dim=8, dropout_rate=0.2):
    model = keras.Sequential([
        keras.layers.Conv2D(filters=output_dim, kernel_size=1, activation=None, padding='same', data_format='channels_first'),
        keras.layers.BatchNormalization(axis=1),
        keras.layers.Activation('gelu'),
        keras.layers.Dropout(dropout_rate)
    ])
    return model

def conv_2d_1x3(hidden_dim=4, output_dim=8, dropout_rate=0.2):
    model = keras.Sequential([
        keras.layers.Conv2D(filters=hidden_dim, kernel_size=1, activation=None, padding='same', data_format='channels_first'),
        keras.layers.BatchNormalization(axis=1),
        keras.layers.Activation('gelu'),
        keras.layers.Dropout(dropout_rate),
        keras.layers.Conv2D(filters=output_dim, kernel_size=3, activation=None, padding='same', data_format='channels_first'),
        keras.layers.BatchNormalization(axis=1),
        keras.layers.Activation('gelu'),
        keras.layers.Dropout(dropout_rate)
    ])
    return model

def conv_2d_1x5(hidden_dim=4, output_dim=8, dropout_rate=0.2):
    model = keras.Sequential([
        keras.layers.Conv2D(filters=hidden_dim, kernel_size=1, activation=None, padding='same', data_format='channels_first'),
        keras.layers.BatchNormalization(axis=1),
        keras.layers.Activation('gelu'),
        keras.layers.Dropout(dropout_rate),
        keras.layers.Conv2D(filters=output_dim, kernel_size=5, activation=None, padding='same', data_format='channels_first'),
        keras.layers.BatchNormalization(axis=1),
        keras.layers.Activation('gelu'),
        keras.layers.Dropout(dropout_rate)
    ])
    return model

def max_pool_2d_to_1x1(output_dim=8, dropout_rate=0.2):
    model = keras.Sequential([
        keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same'),
        keras.layers.Conv2D(filters=output_dim, kernel_size=1, activation=None, padding='same', data_format='channels_first'),
        keras.layers.BatchNormalization(axis=1),
        keras.layers.Activation('gelu'),
        keras.layers.Dropout(dropout_rate)
    ])
    return model
import os
os.environ["KERAS_BACKEND"] = "jax"

import keras

from src.model.layer import gelu_approximate, FeatureWiseScalingLayer
from src.model.model import time_mixer_block


def build_reg_model(input_shape, d_dims=64, dropout_rate=0.2, learning_rate=0.001,
                    mixer_hidden_units=128, feature_pooling='flatten', mixer_input='sequence'):
    """
    누설 유량 예측용 회귀 모델을 만든다.

    Args:
        input_shape: (seq_len, n_features) 형태.
        d_dims (int): conv 트렁크의 채널 수.
        dropout_rate (float): 드롭아웃 비율.
        learning_rate (float): Adam 학습률.
        mixer_hidden_units (int): time_mixer_block 예측 헤드의 은닉층 크기.
            스케일 개수만큼 곱해지므로, 스케일이 많을수록(mixer_input='projection')
            이 값을 줄이는 효과가 커진다. 64 로 낮춰도 성능은 동등했다.
        feature_pooling: conv 트렁크의 출력을 벡터로 만드는 방식.
            'flatten' - 전 시점을 이어붙인다. seq_len x d_dims 크기의 Dense 가 뒤따라
                        파라미터가 크게 늘어난다 (seq_len=30, d_dims=64 에서 172,890개).
            'last'    - 마지막 시점만 쓴다. causal conv 라 마지막 시점이 창 전체를
                        이미 요약하고 있어 파라미터가 크게 줄어든다.
            conv 블록이 conv→norm→activation 으로 바뀐 뒤로는 'last' 가 오히려
            성능을 떨어뜨려(R2 -0.99 → -1.80) 기본값은 'flatten' 이다.
        mixer_input: time_mixer_block 에 무엇을 넣을지.
            'sequence'   - 입력 시계열을 그대로 넣는다. 멀티스케일 분해가 실제 시간 축에
                           적용되므로 블록의 원래 의도에 맞는다. n_features=1 일 때만 쓸 수 있다.
            'projection' - conv 특징을 투영한 pred_len 차원 벡터를 넣는다. 이 벡터는 시간
                           순서가 없는 학습된 특징이라 추세/계절성 분해에 물리적 의미가 없다.
            정확도는 두 방식이 시드 편차 안에서 동률이지만, 'sequence' 가 파라미터가
            26% 적고 블록의 의도에 맞아 기본값이다. 다만 스케일 개수는 입력 길이를 따르므로
            seq_len=30 에서는 2개로 줄어든다 ('projection' 은 pred_len=90 기준 6개).
    """
    if feature_pooling not in ('flatten', 'last'):
        raise ValueError(f"feature_pooling 은 'flatten' 또는 'last' 여야 합니다. (받은 값: {feature_pooling})")
    if mixer_input not in ('sequence', 'projection'):
        raise ValueError(f"mixer_input 은 'sequence' 또는 'projection' 이어야 합니다. (받은 값: {mixer_input})")
    if mixer_input == 'sequence' and input_shape[-1] != 1:
        raise ValueError(f"mixer_input='sequence' 는 단변량 입력(n_features=1)만 지원합니다. "
                         f"(받은 input_shape: {input_shape})")

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

    pred_len = input_shape[0]*3

    if feature_pooling == 'last':
        # causal conv 이므로 마지막 시점이 창 전체를 요약하고 있다.
        y = x_res[:, -1, :]
    else:
        y = keras.layers.Flatten()(x_res)
    y = keras.layers.Dropout(dropout_rate)(y)

    y = FeatureWiseScalingLayer()(y)
    y = keras.layers.Dense(units=pred_len, activation='linear')(y)

    if mixer_input == 'sequence':
        # 실제 시계열에 멀티스케일 분해를 적용한다 (블록의 원래 의도).
        mixer_source = keras.ops.squeeze(input_layer, axis=2)
    else:
        mixer_source = y
    y_res = time_mixer_block(input_layer=mixer_source, pred_len=pred_len,
                             hidden_units=mixer_hidden_units, dropout_rate=dropout_rate)
    y_res = keras.layers.LayerNormalization()(y_res)
    y = y+y_res

    y = FeatureWiseScalingLayer()(y)

    y = keras.layers.Dense(units=1, activation='linear')(y)

    model = keras.models.Model(inputs=input_layer, outputs=y)

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss=keras.losses.LogCosh(),
                  metrics=['mean_absolute_error', 'mean_absolute_percentage_error'])

    return model
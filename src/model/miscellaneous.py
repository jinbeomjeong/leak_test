import os, time
import numpy as np

from scipy.interpolate import CubicSpline
from tqdm.auto import tqdm


def time_warp(series, sigma=0.2, knots=4):
    """
    시계열 데이터에 시간 축 왜곡을 적용합니다.

    Args:
        series (np.array): 1D 시계열 데이터
        sigma (float): 왜곡의 강도를 조절하는 표준편차
        knots (int): 왜곡을 위한 기준점(knot)의 개수

    Returns:
        np.array: 왜곡이 적용된 새로운 시계열 데이터
    """
    time_steps = len(series)

    # 1. 기준점(knot) 선택
    knot_x = np.linspace(0, time_steps - 1, knots)

    # 2. 기준점 왜곡 (랜덤 노이즈 추가)
    knot_y_random = np.random.normal(loc=1.0, scale=sigma, size=(knots,))

    # 3. 부드러운 곡선(Spline) 생성
    # 기준점과 랜덤 노이즈를 곱하여 y축을 왜곡
    spline_func = CubicSpline(knot_x, knot_x * knot_y_random)
    warped_time_index = spline_func(np.arange(time_steps))

    # 4. 데이터 재샘플링 (보간)
    # 원본 데이터의 인덱스와 값, 그리고 왜곡된 시간 인덱스를 사용
    warped_series = np.interp(warped_time_index, np.arange(time_steps), series)

    return warped_series


def create_seq_dataset_trajectory(data: np.array, seq_len=1, horizon=0, target_idx_pos=1):
    """
    미래 구간 전체를 타깃으로 하는 시퀀스 데이터셋을 만듭니다.

    create_seq_dataset_multiple_input_single_output 가 h=horizon 한 점만 타깃으로 삼는 것과 달리,
    h=0..horizon 을 모두 타깃으로 돌려줍니다. 출력을 물리 곡선으로 파라미터화할 때
    (예: level + slope*h) 곡선 파라미터를 여러 점으로 구속하려면 이 형태가 필요합니다.
    한 점만 학습하면 level 과 slope 를 자유롭게 고를 수 있어 파라미터화가 아무 제약도 되지 못합니다.

    h=0 은 입력 창의 마지막 시점과 같은 시각이고, h=horizon 은
    create_seq_dataset_multiple_input_single_output 의 타깃과 같은 시점입니다.
    샘플 개수도 동일해 두 방식을 그대로 비교할 수 있습니다.

    Args:
        data: (n_steps, n_columns) 배열. 0:target_idx_pos 열이 feature, target_idx_pos 열이 target.
        seq_len (int): 입력 창 길이.
        horizon (int): 예측할 미래 길이(스텝).
        target_idx_pos (int): feature 와 target 을 가르는 열 인덱스.

    Returns:
        feature: (n_samples, seq_len, n_features)
        target:  (n_samples, horizon + 1)
    """
    if target_idx_pos < 0:
        raise ValueError(f'target_idx_pos 는 0 이상이어야 합니다. (받은 값: {target_idx_pos})')

    feature, target = [], []

    for i in tqdm(range(data.shape[0] - horizon), desc='creating trajectory dataset...'):
        if i+1 >= seq_len:
            feature.append(data[i+1-seq_len:i+1, 0:target_idx_pos])
            target.append(data[i:i + horizon + 1, target_idx_pos])

    return np.array(feature), np.array(target)


def create_seq_dataset_multiple_input_single_output(data: np.array, seq_len=1, pred_distance=0, target_idx_pos=1):
    # target_idx_pos 는 feature/target 을 가르는 열 인덱스이므로 음수일 수 없다.
    # 기존에는 루프 안에서 target 만 조건부로 append 해 feature 와 개수가 어긋날 수 있었다.
    if target_idx_pos < 0:
        raise ValueError(f'target_idx_pos 는 0 이상이어야 합니다. (받은 값: {target_idx_pos})')

    feature, target = [], []

    for i in tqdm(range(data.shape[0] - pred_distance), desc='creating sequence dataset...'):
        if i+1 >= seq_len:
            feature.append(data[i+1-seq_len:i+1, 0:target_idx_pos])
            target.append(data[i + pred_distance, target_idx_pos:])

    return np.array(feature), np.array(target)  # data shape(n_samples, seq_len, n_features), seq len=[t-29, t-28, t-27,..., t0]
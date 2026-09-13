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


def create_seq_dataset_multiple_input_single_output(data: np.array, seq_len=1, pred_distance=0, target_idx_pos=1):
    feature, target = [], []

    for i in tqdm(range(data.shape[0] - pred_distance), desc='creating sequence dataset...'):
        if i+1 >= seq_len:
            feature.append(data[i+1-seq_len:i+1, 0:target_idx_pos])

            if target_idx_pos >= 0:
                target.append(data[i + pred_distance, target_idx_pos:])

    return np.array(feature), np.array(target)  # data shape(n_samples, seq_len, n_features), seq len=[t-29, t-28, t-27,..., t0]
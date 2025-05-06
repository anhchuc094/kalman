# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
from pykalman import KalmanFilter
from darts import TimeSeries
from darts.models import KalmanFilter as DartsKalman

# Dự báo sử dụng Kalman Basic Model
def kalman_basic(df):
    obs = df['Listening_Time_minutes'].values
    kf = KalmanFilter(
        transition_matrices=[1],
        observation_matrices=[1],
        initial_state_mean=obs[0],
        initial_state_covariance=1,
        observation_covariance=1,
        transition_covariance=0.01
    )
    state_means, _ = kf.filter(obs)
    return state_means

# Kalman Irregular Model
def kalman_irregular(df):
    obs = df['Listening_Time_minutes'].values
    delta_t = np.diff(df['time_num'].values, prepend=df['time_num'].values[0])
    kf = KalmanFilter(
        transition_matrices=np.array([[1]]),
        observation_matrices=np.array([[1]]),
        initial_state_mean=obs[0],
        initial_state_covariance=1,
        observation_covariance=1,
        transition_covariance=0.01
    )
    filtered_state_means = []
    state_mean = obs[0]
    state_cov = 1
    for i in range(len(obs)):
        kf.transition_matrices = np.array([[1 + 0.1 * delta_t[i]]])  # adjust for delta_t
        state_mean, state_cov = kf.filter_update(
            filtered_state_mean=state_mean,
            filtered_state_covariance=state_cov,
            observation=obs[i]
        )
        filtered_state_means.append(state_mean)
    return np.array(filtered_state_means)

# Kalman Filter Darts Model
def kalman_darts(df):
    series = TimeSeries.from_dataframe(df, value_cols="Listening_Time_minutes")
    model = DartsKalman()
    model.fit(series)
    prediction = model.predict(len(series))
    return prediction.values().flatten()

# Dự báo trên tập test (cần tiền xử lý dữ liệu trước)
test_df = pd.read_csv('test.csv')  # Đảm bảo file 'test.csv' có trong thư mục
test_df['Kalman_Basic'] = kalman_basic(test_df)
test_df['Kalman_Irregular'] = kalman_irregular(test_df)
test_df['Kalman_Darts'] = kalman_darts(test_df)

# In kết quả của từng mô hình
print("Kalman Basic Predictions:")
print(test_df[['Podcast_Name', 'Kalman_Basic']])

print("\nKalman Irregular Predictions:")
print(test_df[['Podcast_Name', 'Kalman_Irregular']])

print("\nKalman Darts Predictions:")
print(test_df[['Podcast_Name', 'Kalman_Darts']])

# Lưu kết quả vào tập tin mới
test_df.to_csv("test_with_predictions.csv", index=False)

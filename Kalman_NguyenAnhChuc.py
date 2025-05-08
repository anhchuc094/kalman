# kalman_forecast_wednesday.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.structural import UnobservedComponents

# Step 1: Load and preprocess training data
train_df = pd.read_csv("D:/Chuc/TimeSeries/BTH1/Kalman/train.csv")
df_wed = train_df[train_df['Publication_Day'] == 'Wednesday'].copy()
df_wed.dropna(subset=['Listening_Time_minutes'], inplace=True)  # Drop rows with NaN target
df_wed.reset_index(drop=True, inplace=True)

if len(df_wed) == 0:
    raise ValueError("Không có dữ liệu cho ngày Wednesday sau khi lọc!")

# Target variable: Listening_Time_minutes
y = df_wed['Listening_Time_minutes']

y.index = pd.RangeIndex(start=0, stop=len(y), step=1)

# Step 2: Fit 3 Kalman models
model1 = UnobservedComponents(y, level='local level')
result1 = model1.fit()

model2 = UnobservedComponents(y, level='local linear trend')
result2 = model2.fit()

model3 = UnobservedComponents(y, level='local level', seasonal=7)
result3 = model3.fit()

# Step 3: Evaluate on training set
def calculate_rmse(model_result, name):
    pred = model_result.get_prediction()
    mean_pred = pred.predicted_mean
    rmse = np.sqrt(mean_squared_error(y, mean_pred))
    return {'Model': name, 'RMSE': rmse, 'Prediction': mean_pred}

results = [
    calculate_rmse(result1, 'Local Level'),
    calculate_rmse(result2, 'Local Linear Trend'),
    calculate_rmse(result3, 'Local Level + Seasonality')
]

# Print RMSE comparison
rmse_df = pd.DataFrame(results).drop(columns='Prediction')
print(rmse_df)

# Plot predictions
plt.figure(figsize=(12, 5))
plt.plot(y.values, label='Actual', color='black', linewidth=2)
for res in results:
    plt.plot(res['Prediction'].values, label=f"{res['Model']} (RMSE: {res['RMSE']:.2f})")
plt.title('Comparison of Kalman Models on Training Set')
plt.xlabel('Index')
plt.ylabel('Listening Time')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 4: Forecast on test set
test_df = pd.read_csv("D:/Chuc/TimeSeries/BTH1/Kalman/test.csv")
test_df = test_df[test_df['Publication_Day'] == 'Wednesday'].copy()
test_df.reset_index(drop=True, inplace=True)

if len(test_df) == 0:
    raise ValueError("Không có dữ liệu test cho ngày Wednesday.")
n_forecast = len(test_df)

forecast1 = result1.forecast(steps=n_forecast)
forecast2 = result2.forecast(steps=n_forecast)
forecast3 = result3.forecast(steps=n_forecast)


submission = pd.DataFrame({
    'index': range(len(y), len(y) + n_forecast),
    'forecast_local_level': forecast1,
    'forecast_local_linear': forecast2,
    'forecast_seasonal': forecast3
})
submission.to_csv("D:/Chuc/TimeSeries/BTH1/Kalman/submission_kalman.csv", index=False)
print("Saved forecast to submission_kalman.csv")

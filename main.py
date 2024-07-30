import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

# 예시 데이터 불러오기
data = pd.read_excel('temp.xlsx')  # 데이터 파일 경로에 맞게 수정하세요

# 필요한 열만 선택 (AvgTemp, AvgMeanTemp, AvgMaxTemp)
data = data[['Year', 'AvgTemp', 'AvgMeanTemp', 'AvgMaxTemp']]

# 연도는 제외하고 온도 데이터만 정규화
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[['AvgTemp', 'AvgMeanTemp', 'AvgMaxTemp']])

# 시퀀스 데이터 생성
def create_sequences(data, seq_length):
    x = []
    y = []
    for i in range(len(data) - seq_length):
        x.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(x), np.array(y)

seq_length = 5  # 예시로 5년간의 데이터를 사용
x, y = create_sequences(scaled_data, seq_length)

# 학습 데이터와 테스트 데이터 분리
train_size = int(len(x) * 0.8)
x_train, x_test = x[:train_size], x[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# LSTM 모델 구성
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(seq_length, 3)))
model.add(Dropout(0.2))  # 드롭아웃 레이어 추가
model.add(LSTM(50))
model.add(Dropout(0.2))  # 드롭아웃 레이어 추가
model.add(Dense(3))

model.compile(optimizer='adam', loss='mean_squared_error')
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss')

# 모델 학습
history = model.fit(x_train, y_train, epochs=500, batch_size=16, validation_data=(x_test, y_test),callbacks=[early_stopping, model_checkpoint])

# 마지막 시퀀스를 사용하여 미래 데이터 예측
last_sequence = x_test[-1]

# 향후 5년간의 데이터 예측
future_years = 5
predicted_temps = []

for _ in range(future_years):
    next_pred = model.predict(np.array([last_sequence]))
    predicted_temps.append(next_pred[0])
    last_sequence = np.append(last_sequence[1:], next_pred, axis=0)

predicted_temps = np.array(predicted_temps)
predicted_temps = scaler.inverse_transform(predicted_temps)  # 정규화된 데이터를 원래대로 되돌리기

# 예측 결과를 데이터프레임으로 변환
last_year = data['Year'].iloc[-1]
future_years = np.arange(last_year + 1, last_year + 1 + future_years)
predicted_df = pd.DataFrame(predicted_temps, columns=['PredictedAvgTemp', 'PredictedMeanTemp', 'PredictedMaxTemp'], index=future_years)

print(predicted_df)

# 예측 결과 시각화
plt.plot(data['Year'], data['AvgTemp'], label='Historical AvgTemp')
plt.plot(data['Year'], data['AvgMeanTemp'], label='Historical AvgMeanTemp')
plt.plot(data['Year'], data['AvgMaxTemp'], label='Historical AvgMaxTemp')
plt.plot(predicted_df.index, predicted_df['PredictedAvgTemp'], label='Predicted AvgTemp', linestyle='--')
plt.plot(predicted_df.index, predicted_df['PredictedMeanTemp'], label='Predicted AvgMeanTemp', linestyle='--')
plt.plot(predicted_df.index, predicted_df['PredictedMaxTemp'], label='Predicted AvgMaxTemp', linestyle='--')
plt.xlabel('Year')
plt.ylabel('Temperature')
plt.title('Temperature Prediction')
plt.legend()
plt.show()

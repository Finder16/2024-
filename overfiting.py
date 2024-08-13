import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def lfsr(seed, taps, length):
    state = seed
    n = max(taps) + 1
    stream = []

    for _ in range(length):
        new_bit = 0
        for t in taps:
            new_bit ^= (state >> t) & 1
        state = ((state << 1) & ((1 << n) - 1)) | new_bit
        stream.append(new_bit)

    return stream

# 데이터 생성
num_samples = 10000
lfsr_length = 8
X = []
y = []

fixed_taps = [0, 1, 2, 3]
fixed_seed = 0x1F

for _ in range(num_samples):
    stream = lfsr(fixed_seed, fixed_taps, lfsr_length)

    period = None
    for i in range(1, len(stream) // 2 + 1):
        if stream[:i] == stream[i:2 * i]:
            period = i
            break
    if period is None:
        period = len(stream)

    X.append(stream[:lfsr_length])
    y.append(period)

X = np.array(X)
y = np.array(y)

# 데이터 전처리: X의 모양을 (num_samples, lfsr_length, 1)로 변환
X = X.reshape((num_samples, lfsr_length, 1))

# 데이터 분할: Train/Validation/Test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)  # 70% Train, 30% Temp
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)  # 15% Val, 15% Test

# 하이퍼파라미터 설정 및 모델 구성
units_1 = 128  # 유닛 수 다시 증가
dropout_1 = 0.3  # 드롭아웃 비율 감소
units_2 = 64
dropout_2 = 0.3
units_3 = 64
learning_rate = 1e-4  # 학습률 조정

model = Sequential()
model.add(LSTM(units=units_1, activation='relu', input_shape=(lfsr_length, 1), return_sequences=True, kernel_regularizer=l2(0.001)))
model.add(Dropout(rate=dropout_1))
model.add(GRU(units=units_2, activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dropout(rate=dropout_2))
model.add(Dense(units_3, activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dense(units_3, activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dense(1, activation='linear'))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss='mean_squared_error')

# 조기 종료 콜백 추가
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# 모델 학습
history = model.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val), callbacks=[early_stopping])

# 테스트 데이터에 대한 주기 예측
test_stream = lfsr(fixed_seed, fixed_taps, lfsr_length)
test_stream = np.array(test_stream).reshape((1, lfsr_length, 1))
predicted_period = model.predict(test_stream)[0][0]

print("Generated stream:", test_stream.flatten())
print(f"Predicted Period: {predicted_period}")

# 검증 데이터에 대한 정확도 평가
y_pred_val = model.predict(X_val).flatten()
correct_predictions_val = np.sum(np.abs(y_pred_val - y_val) <= 1)
val_accuracy = correct_predictions_val / len(y_val)
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

# 테스트 데이터에 대한 정확도 평가
y_pred_test = model.predict(X_test).flatten()
correct_predictions_test = np.sum(np.abs(y_pred_test - y_test) <= 1)
test_accuracy = correct_predictions_test / len(y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# 학습 손실 및 검증 손실 시각화
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

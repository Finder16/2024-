import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout, Bidirectional
from sklearn.model_selection import train_test_split
from keras_tuner.tuners import RandomSearch
import matplotlib.pyplot as plt

def lfsr(seed, taps, length):
    state = seed
    n = max(taps) + 1  # 가장 큰 tap 위치에 맞춰 상태의 비트 길이 결정
    stream = []

    for _ in range(length):
        new_bit = 0
        for t in taps:
            new_bit ^= (state >> t) & 1  # 각 tap 위치의 비트를 XOR
        state = ((state << 1) & ((1 << n) - 1)) | new_bit  # 새로운 비트를 상태의 맨 뒤에 추가
        stream.append(new_bit)

    return stream

def generate_lfsr_data(num_samples, lfsr_length):
    X = []
    y = []
    for _ in range(num_samples):
        seed = np.random.randint(1, 2**min(31, lfsr_length))  # 범위를 더 작은 값으로 제한
        taps = np.random.choice(range(lfsr_length), size=np.random.randint(2, lfsr_length), replace=False).tolist()
        stream = lfsr(seed, taps, 2**min(15, lfsr_length))  # 스트림 길이도 제한
        period = find_period(stream)
        X.append(stream[:lfsr_length])
        y.append(period)
    return np.array(X), np.array(y)

def find_period(stream):
    for i in range(1, len(stream)//2 + 1):
        if stream[:i] == stream[i:2*i]:
            return i
    return len(stream)

num_samples = 5000  # 데이터를 더 많이 생성
lfsr_length = 16  # 적절한 값으로 조정
X, y = generate_lfsr_data(num_samples, lfsr_length)

X = X.reshape((num_samples, lfsr_length, 1))  # LSTM/GRU 입력 형태에 맞게 변경
y = np.array(y)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

def build_model(hp):
    model = Sequential()
    model.add(Bidirectional(LSTM(units=hp.Int('units_1', min_value=32, max_value=256, step=32),
                                 activation='relu', input_shape=(lfsr_length, 1), return_sequences=True)))
    model.add(Dropout(rate=hp.Float('dropout_1', min_value=0.1, max_value=0.5, step=0.1)))
    model.add(Bidirectional(GRU(units=hp.Int('units_2', min_value=32, max_value=256, step=32), activation='relu')))
    model.add(Dropout(rate=hp.Float('dropout_2', min_value=0.1, max_value=0.5, step=0.1)))
    model.add(Dense(units=hp.Int('units_3', min_value=32, max_value=256, step=32), activation='relu', 
                    kernel_regularizer=tf.keras.regularizers.l2(hp.Float('l2', min_value=1e-4, max_value=1e-2, sampling='LOG'))))
    model.add(Dense(1, activation='linear'))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                  loss='mean_squared_error')

    return model

tuner = RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=10,  # 시도 횟수
    executions_per_trial=2,  # 각 시도당 실행 횟수
    directory='my_dir',
    project_name='lfsr_tuning'
)

tuner.search(X_train, y_train, epochs=30, validation_data=(X_val, y_val))  # 에포크

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

model = tuner.hypermodel.build(best_hps)
history = model.fit(X_train, y_train, epochs=30, validation_data=(X_val, y_val))  # 에포크

def predict_period(stream):
    stream = np.array(stream).reshape((1, lfsr_length, 1))
    return model.predict(stream)[0][0]

# 임의의 초기값 (128-bit)
seed = 0xDEADBEEFCAFEBABE
taps = [0, 1, 2, 7, 15]  # 최대 lfsr_length - 1 로 설정
length = 16  # lfsr_length와 동일하게 설정

# 스트림 생성
test_stream = lfsr(seed, taps, length)
predicted_period = predict_period(test_stream)
print("Generated stream:", test_stream)
print(f"Predicted Period: {predicted_period}")

# 학습 및 검증 손실 그래프 그리기
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

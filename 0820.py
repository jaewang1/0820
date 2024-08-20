import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

data = pd.read_csv('./data/5.HeightWeight.csv', index_col=0)
# 데이터 준비
data['Height(Inches)'] = data['Height(Inches)'] * 2.54
data['Weight(Pounds)'] = data['Weight(Pounds)'] * 0.453592

array = data.values

X = array[:, 0]
Y = array[:, 1]
# 이차원 행렬
X = X.reshape(-1, 1)

# 데이터 분할
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# 모델 학습
model = LinearRegression()
model.fit(X_train, Y_train)

# 예측

y_prediction = model.predict(X_test)
# 성능 평가
mse = mean_squared_error(Y_test, y_prediction)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_prediction, Y_test)

print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'MAE: {mae}')

plt.figure(figsize=(10, 6))

# 실제 값 산점도
plt.scatter(X_test[:100], y_prediction[:100], color='red')

# 예측 값 선 그래프
plt.scatter(X_test[:100], Y_test[:100], color='blue')

plt.xlabel('Height(cm)')
plt.ylabel('Weight(kg)')
plt.title('HeightWeight')
plt.show()

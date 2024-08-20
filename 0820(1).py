import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

header = ['CRIM', 'ZN', 'INDUS', 'CHAS',
          'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
data = pd.read_csv('./data/3.housing.csv', delim_whitespace=True, names=header)

# 데이터 전처리
array = data.values
# 독립변수 / 종속변수
X = array[:, 0:13]
Y = array[:, 13]

scaler = MinMaxScaler(feature_range=(0, 1))
rescaled_X = scaler.fit_transform(X)

# 학습데이터 / 테스트데이터
X_train, X_test, Y_train, Y_test = train_test_split(rescaled_X, Y, test_size=0.2)


# 모델 선택 및 학습
model = LinearRegression()
model.fit(X_train, Y_train)
y_prediction = model.predict(X_test)
mse = mean_squared_error(Y_test, y_prediction)
mae = mean_absolute_error(Y_test, y_prediction)

print(mse)
print(mae)

fold = KFold(n_splits=5, shuffle=True)
acc = cross_val_score(model, rescaled_X, Y, cv=fold, scoring='neg_mean_squared_error')
mean_score = acc.mean()

# plt.figure(figsize=(10, 6))
plt.scatter(range(len(X_test[:15])), Y_test[:15], color='blue', label='Actual Values', marker='o')
plt.scatter(range(len(Y_test[:15])), y_prediction[:15], color='red', label='Predicted Values', marker='x')

plt.xlabel('xx')
plt.ylabel('yy')
plt.title('housing')
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('/home/atharva/NSTI/CSV_Files/Price.csv')

print(df.info())
print(df.describe())
print(df.head())
print(df.tail())
print(df.columns)
print(df['Price'])
print(df.isnull().mean())

le = LabelEncoder()
df['City'] = le.fit_transform(df['City'])

X = df[['City', 'Area']]
y = df['Price']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print("Predicted Prices:\n", y_pred)
print("Actual Prices:\n", y_test.values)

plt.figure(figsize=(4, 5))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='green', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Price')
plt.grid(True)
plt.tight_layout()
plt.show()

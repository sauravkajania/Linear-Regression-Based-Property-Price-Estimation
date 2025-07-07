# Real Estate Price Prediction

A machine learning project that predicts real estate prices based on city location and property area using Linear Regression.

## Dataset Overview

The dataset contains 15 real estate records with the following features:
- **City**: Location (Mumbai, Delhi, Kolkata)
- **Area**: Property size in square units
- **Price**: Property price in currency units

## Features

- Data preprocessing and exploration
- Label encoding for categorical variables
- Linear regression model training
- Price prediction based on location and area
- Visualization of actual vs predicted prices

## Installation

```bash
pip install pandas matplotlib scikit-learn
```

## Usage

### Load and Explore Data
```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('Price.csv')
df.info()
df.describe()
```

### Data Preprocessing
```python
le = LabelEncoder()
df['City'] = le.fit_transform(df['City'])
```

### Model Training
```python
X = df[['City','Area']]
y = df['Price']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

model = LinearRegression()
model.fit(x_train, y_train)
```

### Prediction and Visualization
```python
y_pred = model.predict(x_test)

plt.figure(figsize=(4,5))
plt.scatter(y_test, y_pred, color='Blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='green', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Price')
plt.grid(True)
plt.show()
```

## Dataset Statistics

- **Total Records**: 15
- **Features**: 3 (City, Area, Price)
- **No Missing Values**: 100% data completeness
- **Area Range**: 650 - 1510 sq units
- **Price Range**: 4500 - 9000 currency units

## Model Performance

The Linear Regression model predicts prices based on:
- City location (encoded as numerical values)
- Property area in square units

Sample predictions show reasonable accuracy for the small dataset size.

## File Structure

```
├── Price.csv                 # Dataset file
├── real_estate_prediction.py # Main prediction script
└── README.md                 # Project documentation
```



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

train_file_path = 'C:\\Users\\p3204160\\Downloads\\house-train.csv'  
housing_train_data = pd.read_csv(train_file_path)

numeric_features = housing_train_data.select_dtypes(include=['int64', 'float64']).drop('SalePrice', axis=1)
target = housing_train_data['SalePrice']
  
imputer = SimpleImputer(strategy='median')
numeric_features_imputed = imputer.fit_transform(numeric_features)

scaler = StandardScaler()
numeric_features_scaled = scaler.fit_transform(numeric_features_imputed)

X_train, X_val, y_train, y_val = train_test_split(numeric_features_scaled, target, test_size=0.2, random_state=42)

model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1)  # Output layer for regression
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32, verbose=0)

plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

val_loss, val_mae = model.evaluate(X_val, y_val, verbose=0)
print(f"Validation Loss: {val_loss}, Validation MAE: {val_mae}")


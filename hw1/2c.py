import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

train_file_path = 'C:\\Users\\p3204160\\Downloads\\house-train.csv'  # Update this to your file path
housing_train_data = pd.read_csv(train_file_path)

X = housing_train_data.drop('SalePrice', axis=1)
y = housing_train_data['SalePrice']

numeric_columns = X.select_dtypes(include=['int64', 'float64']).columns
categorical_columns = X.select_dtypes(include=['object']).columns

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_columns),
    ('cat', categorical_transformer, categorical_columns)
])

X_processed = preprocessor.fit_transform(X)
X_processed = X_processed.toarray()  # Convert to dense array if output is sparse

X_train, X_val, y_train, y_val = train_test_split(X_processed, y, test_size=0.2, random_state=42)

model_complex = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)  # Output layer for regression
])

model_complex.compile(optimizer='adam', loss='mse', metrics=['mae'])

history_complex = model_complex.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32, verbose=0)

plt.figure(figsize=(10, 6))
plt.plot(history_complex.history['loss'], label='Training Loss')
plt.plot(history_complex.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss - More Complex Model')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

val_loss_complex, val_mae_complex = model_complex.evaluate(X_val, y_val, verbose=0)
print(f"Validation Loss: {val_loss_complex}, Validation MAE: {val_mae_complex}")



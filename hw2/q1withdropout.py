import tensorflow as tf
from tensorflow.keras import layers, models, datasets, optimizers

# Define the simplified AlexNet model
def SimplifiedAlexNet(input_shape=(32, 32, 3), num_classes=10):
    model = models.Sequential()
    model.add(layers.Conv2D(48, (5, 5), strides=(1, 1), padding='same', input_shape=input_shape, activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (5, 5), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(192, (3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(192, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

# Load CIFAR-10 data
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

# Initialize the model, compile it with a loss function, optimizer, and evaluation metric
model = SimplifiedAlexNet()
model.compile(optimizer=optimizers.Adam(lr=1e-4),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Display the model's architecture
model.summary()

# Train the model
model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

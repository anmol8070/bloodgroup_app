from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Model architecture
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(8, activation='softmax')  # 8 classes: A+, B+, O+, AB+ (or adjust to your classes)
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Image generator
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    'model/dataset/',

    target_size=(128, 128),
    batch_size=16,
    class_mode='categorical',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    'model/dataset/',
    target_size=(128, 128),
    batch_size=16,
    class_mode='categorical',
    subset='validation'
)

# Train
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)

# Save model
model.save('model/blood_group_cnn.h5') 

print("✅ Model saved as model/blood_group_cnn.h5")

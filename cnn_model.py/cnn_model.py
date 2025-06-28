from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def build_and_train_model():
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(4, activation='softmax')  # Adjust class count based on your blood group classes
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train = datagen.flow_from_directory('dataset/', target_size=(128,128), batch_size=32,
                                        class_mode='categorical', subset='training')
    val = datagen.flow_from_directory('dataset/', target_size=(128,128), batch_size=32,
                                      class_mode='categorical', subset='validation')

    model.fit(train, validation_data=val, epochs=10)

    model.save('model/blood_group_cnn.h5')
    return model

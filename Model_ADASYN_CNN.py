import numpy as np
from imblearn.over_sampling import ADASYN
from keras import Sequential
from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from keras.src.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from Evaluation_nrml import evaluation


def Model_ADASYN_CNN(Train_Data, Train_Target, Test_Data, Test_Target):
    if len(Train_Data.shape) == 3:  # If grayscale, add channel dimension
        Train_Data = Train_Data.reshape(-1, Train_Data.shape[1], Train_Data.shape[2], 1)
        Test_Data = Test_Data.reshape(-1, Test_Data.shape[1], Test_Data.shape[2], 1)

    # Normalize data
    Train_Data, Test_Data = Train_Data / 255.0, Test_Data / 255.0

    # Encode labels if necessary
    encoder = LabelEncoder()
    Train_Target = encoder.fit_transform(Train_Target)
    Test_Target = encoder.transform(Test_Target)

    # Apply ADASYN for class imbalance
    Train_Data_flat = Train_Data.reshape(Train_Data.shape[0], -1)  # Flatten for ADASYN
    adasyn = ADASYN(sampling_strategy='auto', random_state=42)
    Train_Data_resampled, Train_Target_resampled = adasyn.fit_resample(Train_Data_flat, Train_Target)

    # Reshape back to original image dimensions
    Train_Data_resampled = Train_Data_resampled.reshape(-1, Train_Data.shape[1], Train_Data.shape[2], 1)

    # One-hot encoding
    num_classes = len(np.unique(Train_Target))
    Train_Target_resampled = to_categorical(Train_Target_resampled, num_classes)
    Test_Target = to_categorical(Test_Target, num_classes)

    # CNN Model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(Train_Data.shape[1], Train_Data.shape[2], 1)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(Train_Data_resampled, Train_Target_resampled, epochs=10, batch_size=32, validation_data=(Test_Data, Test_Target))

    predictions = model.predict(Test_Data)
    Pred_Mean = np.mean(predictions)
    Pred = np.where(predictions < Pred_Mean, 0, 1)
    Pred = Pred.astype('int')
    Eval = evaluation(Pred, Test_Target)
    return Eval


from keras import Sequential
from keras.src.layers import Dense
from sklearn.preprocessing import LabelEncoder
from Evaluation_nrml import evaluation


def Model_ANN(Train_Data, Train_Target, Test_Data, Test_Target):
    if len(Train_Target.shape) == 1:
        encoder = LabelEncoder()
        Train_Target = encoder.fit_transform(Train_Target)
        Test_Target = encoder.transform(Test_Target)

    # Define ANN model
    model = Sequential([
        Dense(64, activation='relu', input_shape=(Train_Data.shape[1],)),  # Input layer
        Dense(32, activation='relu'),  # Hidden layer
        Dense(Train_Target.shape[1], activation='softmax')  # Output layer
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(Train_Data, Train_Target, epochs=50, batch_size=32, verbose=1, validation_split=0.2)

    # Evaluate the model
    predictions = model.predict(Test_Data)
    Eval = evaluation(predictions, Test_Target)
    return Eval
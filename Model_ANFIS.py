import keras
from keras import Input
from keras.src.layers import Dense
from sklearn.preprocessing import StandardScaler
from Evaluation_nrml import evaluation

def Model_ANFIS(X_train, X_test, y_train, y_test):
    # Normalize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # Define ANFIS-like Model in Keras
    model = keras.Sequential([
        Input(shape=(4,)),  # Input layer (4 features)
        Dense(16, activation='relu'),  # Hidden layer (Fuzzy rule processing)
        Dense(8, activation='relu'),  # Further feature abstraction
        Dense(1, activation='sigmoid')  # Output layer (Binary classification)
    ])
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=1, validation_data=(X_test, y_test))
    # Evaluate the model
    y_pred = model.predict(X_test)
    y_pred_classes = (y_pred > 0.5).astype(int)
    Eval = evaluation(y_pred_classes, y_test)
    return Eval


import numpy as np
from keras import Sequential, models
from keras.src.layers import Dense, Dropout
from keras.src.optimizers import Adam
from sklearn.preprocessing import StandardScaler





def Model_SVM_Feat(Train_Data, Train_Target):
    # # Standardize the data
    scaler = StandardScaler()
    Train_Data = scaler.fit_transform(Train_Data)
    num_classes = Train_Target.shape[1]
    # Define the model
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(Train_Data.shape[1],)))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))  # Multi-class output
    model.compile(optimizer=Adam(learning_rate=0.001),loss='categorical_crossentropy' if num_classes > 1 else 'binary_crossentropy',metrics=['accuracy'])
    # Train the model
    layer_no = 4
    model.fit(Train_Data, Train_Target, batch_size=64, epochs=2)
    intermediate_model = models.Model(inputs=model.inputs, outputs=model.layers[layer_no].output)
    Feats = intermediate_model.get_weights()[layer_no]
    Feats = np.resize(Feats, [Train_Data.shape[0], 100])
    return Feats


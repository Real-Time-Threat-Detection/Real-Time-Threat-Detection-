import keras
from TorchCRF import CRF
from keras import Input
from keras.src.layers import Bidirectional, LSTM, Dense
from Evaluation_nrml import evaluation

def Model_ADeepCRF(Train_Data, Train_Target, Test_Data, Test_Target,sol=None):
    if sol is None:
        sol = [5,0.01,100]
    num_classes = Train_Target.shape[1]  # Number of classes
    input_shape = Train_Data.shape[1:]  # Input shape

    # Define model
    inputs = Input(shape=input_shape)
    x = Bidirectional(LSTM(int(sol[0]), return_sequences=True))(inputs)  # Bi-LSTM layer
    x = Dense(64, activation="relu")(x)
    crf = CRF(num_classes)  # CRF Layer
    outputs = crf(x)

    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", learning_rate=sol[1],loss=crf.loss, metrics=[crf.accuracy])

    # Train the model
    model.fit(Train_Data, Train_Target, batch_size=32, epochs=10,steps_per_epoch=int(sol[2]), validation_data=(Test_Data, Test_Target))
    predictions = model.predict(Test_Data)
    Eval = evaluation(predictions, Test_Target)
    return Eval


from tensorflow.keras.layers import Dense, Input, Activation, Dropout
from tensorflow.python.keras.models import Model


def create_ensemble_model(input_size):
    input = Input((input_size))

    z = Dense(200)(input)
    z = Activation("relu")(z)
    z = Dropout(0.2) (z)
    # z = Dense(60)(input)
    # z = Activation("relu")(z)

    # Softmax to generate a probability distribution.
    z = Dense(30)(z)
    z = Activation("softmax")(z)

    # our model will accept the inputs of the two branches and
    # then output a single value
    return Model(inputs=input, outputs=z)

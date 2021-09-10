import os
import sys
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Dropout, Flatten, Activation, Concatenate, GlobalAveragePooling1D, GlobalMaxPooling1D, GRU
from tensorflow.python.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from rps import rps

sys.path.append(os.path.abspath('./'))
import config


def create_keras_model(input_shape_x, input_shape_t):
    # Define two sets of inputs
    temporal_input = Input(shape=input_shape_x)
    target_input = Input(shape=input_shape_t,)

    # Branch 1: just flatten the target stock input (this is not really needed).
    target_out = Flatten()(target_input)

    # Branch 2: GRU for sequential data
    xt = GRU(16, return_sequences=True, recurrent_dropout=0.1)(temporal_input)
    xt = GRU(16, return_sequences=True, recurrent_dropout=0.1)(xt)
    max_pool = GlobalMaxPooling1D()(xt)
    avg_pool = GlobalAveragePooling1D()(xt)

    # Combine the output of the two branches
    combined = Concatenate()([target_out, max_pool, avg_pool])

    # Output block.
    z = Dense(200)(combined)
    z = Activation("relu")(z)
    z = Dropout(0.3)(z)

    # Softmax to generate a probability distribution.
    z = Dense(30)(z)
    z = Activation("softmax")(z)

    return Model(inputs=[temporal_input, target_input], outputs=z)


class GRUBasedNN():
    def __init__(self):
        self.model = None

    def train(self, X_train, T_train, Y_train, X_test, T_test, Y_test, output_file, batch_size=config.DEFAULT_BATCH_SIZE):
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(
            physical_devices[0], enable=True)

        if self.model == None:
            self.model = create_keras_model(
                (X_train.shape[1], X_train.shape[2]), T_train.shape[1])

        opt = Adam(learning_rate=0.002)
        self.model.compile(loss=rps, optimizer=opt)

        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss', patience=15, verbose=0, mode='min')
        checkpoint = ModelCheckpoint(
            output_file, save_best_only=True, monitor='val_loss', mode='min')
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, verbose=1, epsilon=1e-4, mode='min')

        self.model.fit([X_train, T_train], Y_train, batch_size=batch_size, epochs=1000,
                       verbose=True, callbacks=[checkpoint, reduce_lr, early_stopping], validation_data=([X_test, T_test], Y_test), shuffle=False)

    def load(self, path):
        self.model = tf.keras.models.load_model(path, compile=False)

    def predict(self, X, T):
        return self.model.predict([X, T], batch_size=config.DEFAULT_BATCH_SIZE)

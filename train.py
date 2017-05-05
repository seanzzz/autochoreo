import numpy as np
from sklearn import preprocessing
from python_speech_features import mfcc
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Dropout
from keras.layers import LSTM, GRU, SimpleRNN
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.utils import np_utils


def process_data(dance_data, music_data, steps=120):
    num_frames = dance_data.shape[0]
    x = np.concatenate((dance_data, music_data), axis=1)
    data_in = []
    data_out = []
    for i in range(num_frames - steps):
        data_in.append(x[i:i + steps])
        data_out.append(dance_data[i + steps])

    data_in = np.array(data_in)
    data_out = np.array(data_out)
    print "expected input shape:", data_in.shape
    print "expected output shape:", data_out.shape

    return (data_in, data_out)


def truncate_data(data_in, data_out, batch_size):
    truncate = data_in.shape[0] % batch_size
    if truncate != 0:
        data_in = data_in[0:-truncate]
        data_out = data_out[0:-truncate]
    return (data_in, data_out)


def build_model(num_features, num_outputs, num_cells=256, num_layers=2,
                batch_size=120, steps=120, dropout=0, more_dense=False):
    model = Sequential()

    first_lstm_nr = LSTM(num_cells, batch_input_shape=(batch_size, steps,
                         num_features), return_sequences=False, stateful=True)
    first_lstm = LSTM(num_cells, batch_input_shape=(batch_size, steps,
                      num_features), return_sequences=True, stateful=True)
    last_lstm = LSTM(num_cells, return_sequences=False, stateful=True)

    if more_dense:
        num_layers -= 1

    if num_layers == 1:
        model.add(first_lstm_nr)
        model.add(Dropout(dropout))
    elif num_layers > 1:
        model.add(first_lstm)
        model.add(Dropout(dropout))
        for _ in range(1, num_layers - 1):
            middle_lstm = LSTM(num_cells, return_sequences=True, stateful=True)
            model.add(middle_lstm)
            model.add(Dropout(dropout))
        model.add(last_lstm)
        model.add(Dropout(dropout))

    model.add(Dense(num_cells))
    model.add(Dropout(dropout))
    model.add(Dense(num_outputs))
    model.compile(loss='mse', optimizer='RMSprop')
    print model.summary()
    return model


def build_model_no_stateful(num_features, num_outputs, num_cells=256,
                            num_layers=2, batch_size=120, steps=120, dropout=0):
    model = Sequential()

    first_lstm_nr = LSTM(num_cells, input_shape=(steps,
                         num_features), return_sequences=False)
    first_lstm = LSTM(num_cells, input_shape=(steps,
                      num_features), return_sequences=True)
    last_lstm = LSTM(num_cells, return_sequences=False)

    if num_layers == 1:
        model.add(first_lstm_nr)
        model.add(Dropout(dropout))
    elif num_layers > 1:
        model.add(first_lstm)
        model.add(Dropout(dropout))
        for _ in range(1, num_layers - 1):
            middle_lstm = LSTM(num_cells, return_sequences=True)
            model.add(middle_lstm)
            model.add(Dropout(dropout))
        model.add(last_lstm)
        model.add(Dropout(dropout))

    model.add(Dense(num_outputs))
    model.compile(loss='mse', optimizer='RMSprop')
    print model.summary()
    return model


def build_model_GRU(num_features, num_outputs, num_cells=256, num_layers=2,
                    batch_size=120, steps=120):
    model = Sequential()

    first_gru_nr = GRU(num_cells, batch_input_shape=(batch_size, steps,
                       num_features), return_sequences=False, stateful=True)
    first_gru = GRU(num_cells, batch_input_shape=(batch_size, steps,
                    num_features), return_sequences=True, stateful=True)
    last_gru = GRU(num_cells, return_sequences=False, stateful=True)

    if num_layers == 1:
        model.add(first_gru_nr)
    elif num_layers > 1:
        model.add(first_gru)
        for _ in range(1, num_layers - 1):
            middle_gru = GRU(num_cells, return_sequences=True, stateful=True)
            model.add(middle_gru)
        model.add(last_gru)

    model.add(Dense(num_outputs))
    model.compile(loss='mse', optimizer='RMSprop')
    print model.summary()
    return model


def build_model_RNN(num_features, num_outputs, num_cells=256, num_layers=2,
                    batch_size=120, steps=120):
    model = Sequential()

    first_rnn_nr = SimpleRNN(num_cells, batch_input_shape=(batch_size, steps,
                             num_features), return_sequences=False,
                             stateful=True)
    first_rnn = SimpleRNN(num_cells, batch_input_shape=(batch_size, steps,
                          num_features), return_sequences=True, stateful=True)
    last_rnn = SimpleRNN(num_cells, return_sequences=False, stateful=True)

    if num_layers == 1:
        model.add(first_rnn_nr)
    elif num_layers > 1:
        model.add(first_rnn)
        for _ in range(1, num_layers - 1):
            middle_rnn = SimpleRNN(num_cells, return_sequences=True,
                                   stateful=True)
            model.add(middle_rnn)
        model.add(last_rnn)

    model.add(Dense(num_outputs))
    model.compile(loss='mse', optimizer='RMSprop')

    print model.summary()
    return model


def train(model, data_in, data_out, batch_size=120,
          num_cells=256, num_layers=2, epochs=1000, dropout=0,
          cell_type="lstm", stateful=True, more_dense=False):

    num_features = data_in.shape[2]
    steps = data_in.shape[1]
    num_outputs = data_out.shape[1]
    model_json = model.to_json()
    name = "model-%din%dout%dcell%dlayer%dstep%dbatch%.2f-" % \
        (num_features, num_outputs, num_cells, num_layers, steps, batch_size, dropout) \
        + cell_type

    if not stateful:
        name = name + "-ns"

    if more_dense:
        name = name + "-md"

    print name
    with open(name + ".json", "w") as json_file:
        json_file.write(model_json)

    # write_graph = TensorBoard(log_dir='./logs', histogram_freq=0,
    #                           write_graph=True, write_images=False)

    for i in range(epochs):
        print('Epoch', i)
        model.fit(data_in, data_out, batch_size=batch_size, verbose=2,
                  epochs=1, shuffle=False)
        # model.fit(data_in, data_out, batch_size=batch_size, verbose=1,
        #           epochs=1, shuffle=False, callbacks=[write_graph])
        model.reset_states()
        if i % 100 == 99:
            model.save_weights(name + "%d.h5" % i)
            print "Saving model to disk"


def train2(model, data_in, data_out, batch_size=120,
           num_cells=256, num_layers=2, epochs=1000, dropout=0,
           cell_type="lstm", stateful=True, more_dense=False):

    num_features = data_in.shape[2]
    steps = data_in.shape[1]
    num_outputs = data_out.shape[1]
    model_json = model.to_json()
    name = "model-%din%dout%dcell%dlayer%dstep%dbatch%.2f-" % \
        (num_features, num_outputs, num_cells, num_layers, steps, batch_size, dropout) \
        + cell_type

    if not stateful:
        name = name + "-ns"

    if more_dense:
        name = name + "-md"

    name += "_val"
    print name
    with open(name + ".json", "w") as json_file:
        json_file.write(model_json)

    # write_graph = TensorBoard(log_dir='./logs', histogram_freq=0,
    #                           write_graph=True, write_images=False)

    index = 6000
    data_train = data_in[:index]
    labels_train = data_out[:index]
    data_val = data_in[index:]
    labels_val = data_out[index:]

    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    bst_model_path = name + '.h5'
    model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True,
                                       save_weights_only=True)

    for i in range(epochs):
        print('Epoch', i)
        hist = model.fit(data_train, labels_train,
                         validation_data=(data_val, labels_val), verbose=2,
                         epochs=1, batch_size=batch_size, shuffle=False,
                         callbacks=[early_stopping, model_checkpoint])

        # model.fit(data_in, data_out, batch_size=batch_size, verbose=1,
        #           epochs=1, shuffle=False, callbacks=[write_graph])
        model.reset_states()
        # if i % 100 == 99:
        #     model.save_weights(name + "%d.h5" % i)
        #     print "Saving model to disk"


def exp_steps(dance_data, music_data, steps, batch_size=120):
    (data_in, data_out) = process_data(dance_data, music_data, steps)
    (data_in, data_out) = truncate_data(data_in, data_out, batch_size)
    num_features = data_in.shape[2]
    num_outputs = data_out.shape[1]
    model = build_model(num_features, num_outputs, steps=steps)
    train(model, data_in, data_out)


def exp_layers(dance_data, music_data, layers, num_cells):
    (data_in, data_out) = process_data(dance_data, music_data)
    num_features = data_in.shape[2]
    num_outputs = data_out.shape[1]
    model = build_model(num_features, num_outputs,
                        num_cells=num_cells, num_layers=layers)
    train(model, data_in, data_out, num_cells=num_cells, num_layers=layers)


def exp_dropouts(dance_data, music_data, layers, num_cells, dropout):
    (data_in, data_out) = process_data(dance_data, music_data)
    num_features = data_in.shape[2]
    num_outputs = data_out.shape[1]
    model = build_model(num_features, num_outputs,
                        num_cells=num_cells, num_layers=layers, dropout=dropout)
    train(model, data_in, data_out, num_cells=num_cells, num_layers=layers, dropout=dropout)


def exp_cell_type(dance_data, music_data):
    (data_in, data_out) = process_data(dance_data, music_data)
    num_features = data_in.shape[2]
    num_outputs = data_out.shape[1]

    model = build_model_RNN(num_features, num_outputs)
    train(model, data_in, data_out, cell_type="rnn")

    model = build_model_GRU(num_features, num_outputs)
    train(model, data_in, data_out, cell_type="gru")

    model = build_model(num_features, num_outputs)
    train(model, data_in, data_out, cell_type="lstm")


def exp_steps_2(dance_data, music_data, steps, num_cells=181, num_layers=3,
                dropout=0.2, stateful=True, more_dense=False, batch_size=120):
    (data_in, data_out) = process_data(dance_data, music_data, steps)
    (data_in, data_out) = truncate_data(data_in, data_out, batch_size=steps)
    num_features = data_in.shape[2]
    num_outputs = data_out.shape[1]
    if stateful:
        model = build_model(num_features, num_outputs, steps=steps,
                            batch_size=steps, num_layers=num_layers,
                            num_cells=num_cells, dropout=dropout,
                            more_dense=more_dense)
        batch_size = steps
    else:
        model = build_model_no_stateful(num_features, num_outputs, steps=steps,
                                        batch_size=batch_size, num_layers=num_layers,
                                        num_cells=num_cells, dropout=dropout)
    train(model, data_in, data_out, num_cells=num_cells,
          batch_size=batch_size, num_layers=num_layers, dropout=dropout,
          stateful=stateful, more_dense=more_dense)


dance_data = np.load("dance.npy")
music_data = np.load("mfcc_only_7200.npy")
music_data_full = np.load("mfcc_7200.npy")

if dance_data.shape[0] != music_data.shape[0]:
    print "Dance and Music input have different length. Exiting."
    exit(1)


# exp_cell_type(dance_data, music_data)
# exp_layers(dance_data, music_data, layers=1, num_cells=512)
# exp_layers(dance_data, music_data, layers=2, num_cells=256)
# exp_dropouts(dance_data, music_data, layers=2, num_cells=256, dropout=0.2)
# exp_dropouts(dance_data, music_data, layers=2, num_cells=256, dropout=0.5)
# exp_dropouts(dance_data, music_data, layers=2, num_cells=256)
# exp_layers(dance_data, music_data, layers=3, num_cells=181)
# exp_layers(dance_data, music_data, layers=5, num_cells=128)
# exp_steps(dance_data, music_data, 30)
# exp_steps(dance_data, music_data, 60)
# exp_steps(dance_data, music_data, 120)
# exp_steps(dance_data, music_data, 240)
# exp_steps_2(dance_data, music_data, steps=120)
# exp_steps_2(dance_data, music_data_full, steps=120)
# exp_steps_2(dance_data, music_data, steps=120, stateful=False)
# exp_steps_2(dance_data, music_data, num_cells=256, steps=120, more_dense=True)
# exp_steps_2(dance_data, music_data, steps=120, dropout=0.5)
# exp_steps_2(dance_data, music_data, steps=30)
# exp_steps_2(dance_data, music_data, steps=60)
# exp_steps_2(dance_data, music_data, steps=240)

# exp_steps_2(dance_data, music_data, steps=120, batch_size=120, stateful=True)
exp_steps_2(dance_data, music_data, num_cells=256, num_layers=2, steps=30, batch_size=120, dropout=0, stateful=False)
exp_steps_2(dance_data, music_data, num_cells=256, num_layers=2, steps=60, batch_size=120, dropout=0, stateful=False)
exp_steps_2(dance_data, music_data, num_cells=256, num_layers=2, steps=240, batch_size=120, dropout=0, stateful=False)
exp_steps_2(dance_data, music_data, steps=30, batch_size=120, dropout=0.2, stateful=False)
exp_steps_2(dance_data, music_data, steps=60, batch_size=120, dropout=0.2, stateful=False)
exp_steps_2(dance_data, music_data, steps=240, batch_size=120, dropout=0.2, stateful=False)
exp_steps_2(dance_data, music_data, steps=120, batch_size=120, dropout=0.2, stateful=False)
exp_steps_2(dance_data, music_data, num_cells=256, num_layers=2, steps=30, batch_size=120, dropout=0, stateful=False)
# exp_steps_2(dance_data, music_data, num_cells=512, steps=120, batch_size=295, dropout=0, stateful=False)
# exp_steps_2(dance_data, music_data, steps=120, stateful=False)
# exp_steps_2(dance_data, music_data, steps=120, stateful=False)

# epochs = 1000
# batch_size = 120
# step = 120

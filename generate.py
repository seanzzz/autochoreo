import numpy as np

# import scipy.io.wavfile as wav
from python_speech_features import mfcc
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Activation
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils, plot_model


def load_model(exp_name):
    json_file = open(exp_name + ".json", "r")
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(exp_name + "999.h5")
    print "Loaded model from weights"

    plot_model(model, show_shapes=True)
    model.compile(loss='mse', optimizer='RMSprop')
    return model


def generate(model, dance_data, music_data,
             epochs=2000, steps=120, batch_size=120, offset=0):
    num_features = dance_data.shape[1] + music_data.shape[1]
    num_outputs = dance_data.shape[1]
    data_in = []
    data_in = np.zeros((batch_size, steps, num_features))

    # write the first num steps to data_in
    for i in range(steps):
        for j in range(num_outputs):
            data_in[0, i, j] = dance_data[i][j]
        for j in range(num_outputs, num_features):
            data_in[0, i, j] = music_data[i][j - num_outputs]

    print "data_in shape", data_in.shape
    print "data first element:", data_in[0]
    print "data first element:", data_in[0][0]

    output = dance_data[:steps].tolist()
    for i in range(epochs):
        print i
        # gen_in = np.reshape(gen_in, (-1, 120, 43))
        prediction = model.predict(data_in, batch_size=batch_size, verbose=0)

        # update data_in that appends the first output to the last
        for k in range(steps - 1):
            for j in range(num_features):
                data_in[0, k, j] = data_in[0, k + 1, j]

        first_prediction = prediction[0]
        # print first_prediction
        # exit()
        # attach the new one to the output
        for j in range(num_outputs):
            data_in[0, steps - 1, j] = first_prediction[j]

        # update the music feature
        for j in range(num_outputs, num_features):
            data_in[0, steps - 1, j] = \
                music_data[i + offset + 1][j - num_outputs]

        output.append(first_prediction)
    output = np.array(output)
    return output


num_frames = 7200


dance_data = np.load("dance.npy")
music_data = np.load("mfcc_only_7200.npy")


batch_size = 120
steps = 120
epochs = 3000

# exp_name = "model-43in30out256cell2layer120step-rnn"
# model = load_model(exp_name)
# output = generate(model, dance_data, music_data, steps=steps, offset=steps, epochs=6800)
# np.save(exp_name + "_output_complete_long", output)

# exp_name = "model-43in30out256cell2layer120step-gru"
# model = load_model(exp_name)
# output = generate(model, dance_data, music_data, steps=steps, offset=steps, epochs=6800)
# np.save(exp_name + "_output_complete_long", output)

exp_name = "model-43in30out-512cells1layers120steps"
model = load_model(exp_name)
output = generate(model, dance_data, music_data, steps=steps, offset=steps, epochs=6800)
np.save(exp_name + "_output_complete_long", output)

exp_name = "model-43in30out-181cells3layers120steps"
model = load_model(exp_name)
output = generate(model, dance_data, music_data, steps=steps, offset=steps, epochs=6800)
np.save(exp_name + "_output_complete_long", output)

exp_name = "model-43in30out128cell5layer120step-lstm"
model = load_model(exp_name)
output = generate(model, dance_data, music_data, steps=steps, offset=steps, epochs=6800)
np.save(exp_name + "_output_complete_long", output)

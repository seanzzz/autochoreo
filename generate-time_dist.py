import numpy as np

# import scipy.io.wavfile as wav
from python_speech_features import mfcc
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Activation
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

num_frames = 7200


mfcc_feat = np.load("mfcc_7200.npy")
a = np.load("dance.npy")

print a[0]
print mfcc_feat[0]
x = np.concatenate((a, mfcc_feat), axis=1)

# x = x[720:]
print "x shape:", x.shape

batch_size = 120
step = 120
num_features = mfcc_feat.shape[1]
num_outputs = a.shape[1]


data_in = []
data_out = []

data_in = np.zeros((batch_size, step, num_features))

# write the first 120 steps to data_in
for i in range(step):
    for j in range(num_features):
        data_in[0, i, j] = mfcc_feat[i][j]

print "data_in shape", data_in.shape
print "data first element:", data_in[0]
print "data first element:", data_in[0][0]
# data_in = np.array(data_in)
# data_out = np.array(data_out)

# exit()

epochs = 2000
json_file = open("model-69output-time-dist-999.json", "r")
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("model-69output-time-dist-999.h5")
print "Loaded model from weights"

model.compile(loss='mse', optimizer='RMSprop')

print "Input shape,", data_in.shape

# data_in = data_in[720:840]
output = []
for i in range(1, 60):
    print i
    # gen_in = np.reshape(gen_in, (-1, 120, 43))
    prediction = model.predict(data_in, batch_size=batch_size, verbose=0)

    # update data_in that appends the first output to the last
    for k in range(step - 1):
        for j in range(num_features):
            data_in[0, k, j] = mfcc_feat[i * step + k, j]

    output.append(prediction[0])

    # first_prediction = prediction[0]
    # print first_prediction
    # exit()
    # attach the new one to the output
    # for j in range(num_outputs):
    #     data_in[0, step - 1, j] = first_prediction[j]

    # # update the music feature
    # for j in range(num_outputs, num_features):
    #     data_in[0, step - 1, j] = x[i + 3500 + 1][j]

    # output.append(first_prediction)


output = np.array(output)

np.save("output-30out-time-dist", output)

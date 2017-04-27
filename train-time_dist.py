import numpy as np
from sklearn import preprocessing
from python_speech_features import mfcc
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Dropout
from keras.layers import LSTM, TimeDistributed
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.utils import np_utils

num_frames = 7200

min_max_scaler = preprocessing.MinMaxScaler()

mfcc_feat = np.load("mfcc_7200.npy")

a = np.load("dance.npy")

x = np.concatenate((a, mfcc_feat), axis=1)


print x.shape

num_features = mfcc_feat.shape[1]
num_outputs = a.shape[1]
batch_size = 120
step = 120

data_in = []
data_out = []
for i in range(num_frames - step):
    data_in.append(mfcc_feat[i:i + step])
    data_out.append(a[i:i + step])


data_in = np.array(data_in)
data_out = np.array(data_out)

print data_in
print data_out
print "expected input shape:", data_in.shape
print "expected output shape:", data_out.shape
print num_features

# exit()

epochs = 1000
model = Sequential()
model.add(TimeDistributed(Dense(256), input_shape=(step,num_features)))
# model.add(LSTM(1024, return_sequences=True, stateful=True))
model.add(TimeDistributed(Dense(256)))
model.add(TimeDistributed(Dense(num_outputs)))
model.compile(loss='mse', optimizer='RMSprop')

# write_graph = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True,
#                           write_images=False)

# model.fit(data_in, data_out, batch_size=batch_size, verbose=1,
#           epochs=1, shuffle=False, callbacks=[write_graph])

for i in range(epochs):
    print('Epoch', i)
    model.fit(data_in, data_out, batch_size=batch_size, verbose=2,
              epochs=1, shuffle=False)
    model.reset_states()
    if i % 100 == 99:
        model_json = model.to_json()
        with open("model-69output-time-dist-%d.json" % i, "w") as json_file:
            json_file.write(model_json)
        model.save_weights("model-69output-time-dist-%d.h5" % i)
        print "Saving model to disk"

import pandas as pd
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from matplotlib import pyplot

SIZE_OF_SEQUENCE = 36


def time_before(last_event, current_event):
    difference = current_event - last_event
    return difference.seconds


def split_sequence(PDList, n_steps):
    X, y = list(), list()
    for j in PDList:
        sequence = j['value'].tolist()
        for i in range(len(sequence)):
            # find the end of this pattern
            end_ix = i + n_steps
            # check if we are beyond the sequence
            if end_ix > len(sequence) - 1:
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
            X.append(seq_x)
            y.append(seq_y)
    return array(X), array(y)


def get_data(filename):
    data = pd.read_csv(filename, delimiter=";")
    data.rename(columns={
        'DAY': 'day',
        'TIME': 'time',
        'UDT_CGMS': 'value'
    }, inplace=True)
    data['moment'] = data['day'].map(str) + ' ' + data['time'].map(str)
    data['moment'] = pd.to_datetime(data.moment)
    data = data[['moment', 'value']]
    data = data.dropna(subset=['value'])
    return data


def sanitaze_data(data, SIZE_OF_SEQUENCE):
    datas = []
    start = 0
    for index, row in data.iterrows():
        if index != 0:
            data.loc[index, 'previous'] = time_before(data.loc[index - 1, 'moment'], data.loc[index, 'moment'])
        if index < data.shape[0] - 1:
            data.loc[index, 'next'] = time_before(data.loc[index, 'moment'], data.loc[index + 1, 'moment'])
        if data.loc[index, 'next'] > 370:
            # print (data.loc[index])
            temp = data[start:index]
            temp.reset_index(drop=True, inplace=True)
            start = index + 1
            datas.append(temp)

    newlist = []
    for i in datas:
        if i.shape[0] > SIZE_OF_SEQUENCE:
            newlist.append(i)
    return newlist


input = get_data("data.csv")
input = sanitaze_data(input, SIZE_OF_SEQUENCE)

X, y=split_sequence(input,SIZE_OF_SEQUENCE)
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))
# define model
model = Sequential()
model.add(LSTM(SIZE_OF_SEQUENCE*3, activation='relu', input_shape=(SIZE_OF_SEQUENCE, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
history=model.fit(X, y, validation_split=0.33, epochs=300, verbose=2)
print(history.history['loss'])
print(history.history['val_loss'])
pyplot.plot(history.history['loss'])
pyplot.plot(history.history['val_loss'])
pyplot.title('model train vs validation loss')
pyplot.ylabel('loss')
pyplot.xlabel('epoch')
pyplot.legend(['train', 'validation'], loc='upper right')
pyplot.show()

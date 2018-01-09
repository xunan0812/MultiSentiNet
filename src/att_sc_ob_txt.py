#coding=utf8
import numpy as np
np.random.seed(1337)  # for reproducibility
import re
import h5py
import os
from nltk import tokenize
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from attention import Attention_input1, Attention_input2
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Reshape, Dense, Input, Flatten, Dropout, merge, BatchNormalization
from keras.layers import TimeDistributed, LSTM, GRU, Bidirectional
from keras.models import Model
from keras.optimizers import SGD, Adadelta, Adam, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.core import Reshape, RepeatVector

from keras.callbacks import EarlyStopping
from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Input, Merge, Convolution1D, MaxPooling1D


GLOVE_DIR = '../data/'



MAX_SEQUENCE_LENGTH = 140
MAX_NB_WORDS = 10000
EMBEDDING_DIM = 200
VALIDATION_SPLIT = 0.1
NB_EPOCH = 100
NB_CLASS = 3
DIM_HIDDEN = 128
DIM_LSTM = 128


# datamode = 'mul'
datamode = 'single'

if datamode == 'mul':
    DATA_PATH = '../data/MSVA_multiple_17024.h5'
    BATCH_SIZE = 128
else:
    DATA_PATH = '../data/MSVA_single_4511.h5'
    BATCH_SIZE = 32


def load_data():
    read_file = h5py.File(DATA_PATH, 'r')
    texts = read_file['txt_data'][:]
    labels = read_file['label'][:]
    scenes = read_file['scene_data'][:]
    objects = read_file['object_data'][:]
    return texts,labels,scenes,objects
def split_data(data,VALIDATION_SPLIT):
    nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
    data_train = data[:-(nb_validation_samples * 2)]
    data_val = data[-(nb_validation_samples * 2):-(nb_validation_samples)]
    data_test = data[-nb_validation_samples:]
    return data_train,data_val,data_test

def dp_txt(txt):
    # nonEnglish_regex = re.compile('[^a-zA-Z0-9\\?\\!\\,\\.@#\\+\\-=\\*\'\"><&\\$%\\(\\)\\[\\]:;]+')
    hashtag_pattern = re.compile('#[a-zA-Z0-9]+')
    at_pattern = re.compile('@[a-zA-Z0-9]+')
    http_pattern = re.compile("((http|ftp|https)://)(([a-zA-Z0-9\._-]+\.[a-zA-Z]{2,6})|([0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}))(:[0-9]{1,4})*(/[a-zA-Z0-9\&%_\./-~-]*)?")
    txt = txt.strip()
    txt_hashtag = re.sub(hashtag_pattern, '', txt)
    txt_nonat = re.sub(at_pattern, '', txt_hashtag)
    txt_nonhttp = re.sub(http_pattern, '', txt_nonat)
    txt = txt_nonhttp
    return txt


def fun():
    texts,labels,scenes,objects = load_data()
    new_texts = []
    for idx in range(len(texts)):
        text = texts[idx]
        text = dp_txt(str(text))
        new_texts.append(text)
    texts = new_texts

    tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index


    text_data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    labels = to_categorical(np.asarray(labels))

    # print('Text tensor shape:', text_data.shape)
    # print('Label tensor shape:', labels.shape)
    # print('Scene tensor shape:', scenes.shape)
    # print('Object tensor shape:', objects.shape)
    # # split the text_data into a training set and a validation set
    rand = np.arange(labels.shape[0])
    np.random.shuffle(rand)
    indices = rand

    text_data = text_data[indices]
    labels = labels[indices]
    scenes = scenes[indices]
    objects = objects[indices]
    text_train,text_val,text_test = split_data(text_data,VALIDATION_SPLIT)
    label_train,label_val,label_test = split_data(labels,VALIDATION_SPLIT)
    scene_train,scene_val,scene_test = split_data(scenes,VALIDATION_SPLIT)
    object_train,object_val,object_test = split_data(objects,VALIDATION_SPLIT)

    text_shape = text_train.shape[1:]
    scene_shape = scene_train.shape[1:]
    object_shape = object_train.shape[1:]

    embeddings_index = {}
    f = open(os.path.join(GLOVE_DIR, 'glove.6B.200d.txt'))
    for line in f:
        values = line.split()
        word = values[2]
        coefs = np.asarray(values[1], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    nb_words = min(MAX_NB_WORDS, len(word_index))
    embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i > MAX_NB_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    embedding_layer = Embedding(nb_words + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True)

    save_best = ModelCheckpoint('../../model/{}.hdf5'.format('my_weight'), save_best_only=True)
    elstop = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=5)


    # Image Sence
    scene_input = Input(shape=scene_shape, dtype='float32')
    img_scene = Dense(DIM_HIDDEN, activation='relu')(scene_input)
    img_scene_encoder = RepeatVector(text_shape[0], name='scene-repeat')(img_scene)
    
    # Image Object
    object_input = Input(shape=object_shape, dtype='float32')
    img_object = Dense(DIM_HIDDEN, activation='relu')(object_input)
    img_object_encoder = RepeatVector(text_shape[0], name='object-repeat')(img_object)
    
    # Text
    txt_input = Input(shape=text_shape, dtype='float32')
    txt = embedding_layer(txt_input)
    txt_hidden = (LSTM(DIM_HIDDEN, return_sequences=True, name='tweet-lstm'))(txt)
    txt_att = Attention_input2(name='att_so')([txt_hidden, img_object_encoder, img_scene_encoder])

    # Merge
    img_txt = merge([img_scene, img_object, txt_att], mode='concat')
    img_txt = Dense(DIM_HIDDEN, activation='relu')(img_txt)
    img_txt_loss = Dense(NB_CLASS, activation='softmax', name='main_output')(img_txt)

    model = Model(input=[txt_input, scene_input, object_input], output=[img_txt_loss])
    model.compile(loss='categorical_crossentropy', optimizer='RMSprop',
                  metrics=['acc', 'fmeasure'])
    model.fit([text_train, scene_train, object_train], [label_train],
              validation_data=([text_val, scene_val, object_val], [label_val]),
              nb_epoch=NB_EPOCH, batch_size=BATCH_SIZE, callbacks=[elstop,save_best], verbose=1)

    model.load_weights('../../model/{}.hdf5'.format('my_weight'))
    score = model.evaluate([text_test, scene_test, object_test], label_test, verbose=0)


    print('results：', score[1], score[2])
    return score[1:]

if __name__ == '__main__':
    fun()




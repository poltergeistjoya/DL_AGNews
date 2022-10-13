#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sys
import pandas as pd

from absl import flags
from tqdm import tqdm #make progress bar

from transformers import DistilBertTokenizer, DistilBertConfig, TFDistilBertModel

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, precision_score, recall_score

from dataclasses import dataclass, field, InitVar
from joblib import Memory

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Embedding, LSTM, Conv2D, Conv1D, MaxPooling1D, Dense, Dropout, GlobalMaxPooling1D, Input, Bidirectional, concatenate, Flatten, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import plot_model

memory = Memory(".cache")

#Model and tokenizer was implemented with:
#https://www.kaggle.com/code/atechnohazard/news-classification-using-huggingface-distilbert

FLAGS = flags.FLAGS
flags.DEFINE_integer("num_samples", 50000, "Number of samples in dataset")
flags.DEFINE_integer("batch_size", 1024, "Number of samples in batch")
flags.DEFINE_integer("num_iters", 5000, "Number of forward/backward pass iterations")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate/initial step size")
flags.DEFINE_integer("random_seed", 31415, "Random seed for reproducible results")
flags.DEFINE_float("sigma_noise", 0.5, "Standard deviation of noise random variable")
flags.DEFINE_integer("epochs", 5, "Number of epochs")

vocab_size = 20000
embed_size = 32
distil_bert = 'distilbert-base-uncased'

@memory.cache()
def tokenize(sentences, tokenizer, maxlen):
    input_ids, input_masks, input_segments = [],[],[]
    for sentence in tqdm(sentences):
        inputs = tokenizer.encode_plus(sentence, add_special_tokens=True, max_length=maxlen, pad_to_max_length=True,
                                             return_attention_mask=True, return_token_type_ids=True)
        input_ids.append(inputs['input_ids'])
        input_masks.append(inputs['attention_mask'])
        input_segments.append(inputs['token_type_ids'])

    return np.asarray(input_ids, dtype='int32'), np.asarray(input_masks, dtype='int32')

@memory.cache()
def Model(maxlen, batch_size, epochs, train, train_lab, val, val_lab):
    config = DistilBertConfig(dropout=0.2, attention_dropout=0.2)
    config.output_hidden_states = False
    transformer_model = TFDistilBertModel.from_pretrained(distil_bert, config=config)

    input_ids_in = tf.keras.layers.Input(shape=(maxlen,), name='input_token', dtype='int32')
    input_masks_in = tf.keras.layers.Input(shape=(maxlen,), name='masked_token', dtype='int32')

    embedding_layer = transformer_model(input_ids_in, attention_mask=input_masks_in)[0]
    X = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(embedding_layer)
    X = tf.keras.layers.GlobalMaxPool1D()(X)
    X = tf.keras.layers.Dense(64, activation='relu')(X)
    X = tf.keras.layers.Dropout(0.2)(X)
    X = tf.keras.layers.Dense(4, activation='sigmoid')(X)
    model = tf.keras.Model(inputs=[input_ids_in, input_masks_in], outputs = X)

    for layer in model.layers[:3]:
        layer.trainable = False

    model.summary()
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    model.fit(train, train_lab, epochs=epochs, batch_size=batch_size, validation_data=(val, val_lab))

    return model

def main():

    #parse flags before we use them
    FLAGS(sys.argv)
    epochs = FLAGS.epochs
    batch_size = FLAGS.batch_size

    #set seed for reproducible results
    seed_sequence = np.random.SeedSequence(FLAGS.random_seed)
    np_seed, tf_seed = seed_sequence.spawn(2) #spawn 2 sequences for 2 threads
    np_rng =np.random.default_rng(np_seed)

    TRAIN_FILE_PATH = './train.csv'
    TEST_FILE_PATH = './test.csv'

    data =pd.read_csv(TRAIN_FILE_PATH)
    testdata = pd.read_csv(TEST_FILE_PATH)

    X = data['Title'] + " " + data['Description']
    Y = data['Class Index'].apply(lambda x: x-1)

    X_train = X.iloc[:100000]
    Y_train = Y.iloc[:100000].values

    X_val = X.iloc[100000:]
    Y_val = Y.iloc[100000:].values

    x_test = testdata['Title'] + " " + testdata['Description']
    y_test = testdata['Class Index'].apply(lambda x: x-1).values

    maxlen = X_train.map(lambda x:len(x.split())).max()
    data.describe()

    tokenizer = DistilBertTokenizer.from_pretrained(distil_bert, do_lower_case = True, add_special_tokens=True, max_length = maxlen, pad_to_max_length=True)

    X_train = tokenize(X_train, tokenizer, maxlen)
    X_val = tokenize(X_val, tokenizer, maxlen)
    x_test = tokenize(x_test, tokenizer, maxlen)

    model = Model(maxlen, batch_size, epochs, X_train, Y_train, X_val, Y_val)
    test_loss, test_acc = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=2, return_dict=True)

    #PLOTTING
    #plt.plot(model.history['accuracy'], label='accuracy')
    #plt.plot(model.history['val_accuracy'], label = 'val_accuracy')
    #plt.xlabel('Epoch')
    #plt.ylabel('Accuracy')
    #plt.ylim([0.01, 1])
    #plt.legend(loc='lower right')
    #plt.tight_layout()
    #plt.savefig('./epochaccuracy.pdf')

if __name__ == "__main__":
    main()

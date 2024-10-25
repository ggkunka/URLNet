import re
import time
import datetime
import os
import pdb
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from bisect import bisect_left
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from TextCNN import TextCNN
from utils import *

parser = argparse.ArgumentParser(description="Train URLNet model")

# Data arguments
default_max_len_words = 200
parser.add_argument('--data.max_len_words', type=int, default=default_max_len_words, metavar="MLW",
                    help="Maximum length of URL in words (default: {})".format(default_max_len_words))
default_max_len_chars = 200
parser.add_argument('--data.max_len_chars', type=int, default=default_max_len_chars, metavar="MLC",
                    help="Maximum length of URL in characters (default: {})".format(default_max_len_chars))
default_max_len_subwords = 20
parser.add_argument('--data.max_len_subwords', type=int, default=default_max_len_subwords, metavar="MLSW",
                    help="Maximum length of word in subwords/characters (default: {})".format(default_max_len_subwords))
default_min_word_freq = 1
parser.add_argument('--data.min_word_freq', type=int, default=default_min_word_freq, metavar="MWF",
                    help="Minimum frequency of word in training population to build vocabulary (default: {})".format(default_min_word_freq))
default_dev_pct = 0.001
parser.add_argument('--data.dev_pct', type=float, default=default_dev_pct, metavar="DEVPCT",
                    help="Percentage of training set used for dev (default: {})".format(default_dev_pct))
parser.add_argument('--data.data_dir', type=str, default='train_10000.txt', metavar="DATADIR",
                    help="Location of data file")
default_delimit_mode = 1
parser.add_argument("--data.delimit_mode", type=int, default=default_delimit_mode, metavar="DLMODE",
                    help="0: delimit by special chars, 1: delimit by special chars + each char as a word (default: {})".format(default_delimit_mode))

# Model arguments
default_emb_dim = 32
parser.add_argument('--model.emb_dim', type=int, default=default_emb_dim, metavar="EMBDIM",
                    help="Embedding dimension size (default: {})".format(default_emb_dim))
default_filter_sizes = "3,4,5,6"
parser.add_argument('--model.filter_sizes', type=str, default=default_filter_sizes, metavar="FILTERSIZES",
                    help="Filter sizes of the convolution layer (default: {})".format(default_filter_sizes))
default_emb_mode = 1
parser.add_argument('--model.emb_mode', type=int, default=default_emb_mode, metavar="EMBMODE",
                    help="1: charCNN, 2: wordCNN, 3: char + wordCNN, 4: char-level wordCNN, 5: char + char-level wordCNN (default: {})".format(default_emb_mode))

# Training arguments
default_nb_epochs = 5
parser.add_argument('--train.nb_epochs', type=int, default=default_nb_epochs, metavar="NEPOCHS",
                    help="Number of training epochs (default: {})".format(default_nb_epochs))
default_batch_size = 128
parser.add_argument('--train.batch_size', type=int, default=default_batch_size, metavar="BATCHSIZE",
                    help="Size of each training batch (default: {})".format(default_batch_size))
parser.add_argument('--train.l2_reg_lambda', type=float, default=0.0, metavar="L2LREGLAMBDA",
                    help="L2 lambda for regularization (default: 0.0)")
default_lr = 0.001
parser.add_argument('--train.lr', type=float, default=default_lr, metavar="LR",
                    help="Learning rate for optimizer (default: {})".format(default_lr))

# Logging arguments
parser.add_argument('--log.output_dir', type=str, default="runs/10000/", metavar="OUTPUTDIR",
                    help="Directory of the output model")
parser.add_argument('--log.print_every', type=int, default=50, metavar="PRINTEVERY",
                    help="Print training result every this number of steps (default: 50)")
parser.add_argument('--log.eval_every', type=int, default=500, metavar="EVALEVERY",
                    help="Evaluate the model every this number of steps (default: 500)")
parser.add_argument('--log.checkpoint_every', type=int, default=500, metavar="CHECKPOINTEVERY",
                    help="Save a model every this number of steps (default: 500)")

FLAGS = vars(parser.parse_args())

for key, val in FLAGS.items():
    print("{}={}".format(key, val))

# Data preparation
urls, labels = read_data(FLAGS["data.data_dir"])

high_freq_words = None
if FLAGS["data.min_word_freq"] > 0:
    x1, word_reverse_dict = get_word_vocab(urls, FLAGS["data.max_len_words"], FLAGS["data.min_word_freq"])
    high_freq_words = sorted(list(word_reverse_dict.values()))
    print("Number of words with freq >= {}: {}".format(FLAGS["data.min_word_freq"], len(high_freq_words)))

x, word_reverse_dict = get_word_vocab(urls, FLAGS["data.max_len_words"])
word_x = get_words(x, word_reverse_dict, FLAGS["data.delimit_mode"], urls)
ngramed_id_x, ngrams_dict, worded_id_x, words_dict = ngram_id_x(word_x, FLAGS["data.max_len_subwords"], high_freq_words)

chars_dict = ngrams_dict
chared_id_x = char_id_x(urls, chars_dict, FLAGS["data.max_len_chars"])

pos_x = []
neg_x = []
for i in range(len(labels)):
    label = labels[i]
    if label == 1:
        pos_x.append(i)
    else:
        neg_x.append(i)
print("Overall Mal/Ben split: {}/{}".format(len(pos_x), len(neg_x)))
pos_x = np.array(pos_x)
neg_x = np.array(neg_x)

x_train, y_train, x_test, y_test = prep_train_test(pos_x, neg_x, FLAGS["data.dev_pct"])

x_train_char = get_ngramed_id_x(x_train, ngramed_id_x)
x_test_char = get_ngramed_id_x(x_test, ngramed_id_x)

x_train_word = get_ngramed_id_x(x_train, worded_id_x)
x_test_word = get_ngramed_id_x(x_test, worded_id_x)

x_train_char_seq = get_ngramed_id_x(x_train, chared_id_x)
x_test_char_seq = get_ngramed_id_x(x_test, chared_id_x)

# Convert labels to numpy arrays
y_train = np.array(y_train)
y_test = np.array(y_test)

###################################### Training #########################################################

def train_dev_step(model, x, y, emb_mode, optimizer, is_train=True):
    if is_train:
        dropout_rate = 0.5
    else:
        dropout_rate = 0.0  # No dropout during evaluation

    with tf.GradientTape() as tape:
        # Forward pass
        if emb_mode == 1:
            logits = model(x_char_seq=x[0], training=is_train, dropout_rate=dropout_rate)
        elif emb_mode == 2:
            logits = model(x_word=x[0], training=is_train, dropout_rate=dropout_rate)
        elif emb_mode == 3:
            logits = model(x_char_seq=x[0], x_word=x[1], training=is_train, dropout_rate=dropout_rate)
        elif emb_mode == 4:
            logits = model(x_word=x[0], x_char=x[1], x_char_pad_idx=x[2], training=is_train, dropout_rate=dropout_rate)
        elif emb_mode == 5:
            logits = model(x_char_seq=x[0], x_word=x[1], x_char=x[2], x_char_pad_idx=x[3], training=is_train, dropout_rate=dropout_rate)
        else:
            raise ValueError("Invalid emb_mode: {}".format(emb_mode))

        # Compute loss
        loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        loss = loss_fn(y, logits)

    if is_train:
        # Backward pass and optimization
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # Compute accuracy
    predictions = tf.cast(tf.sigmoid(logits) > 0.5, tf.float32)
    correct_predictions = tf.equal(predictions, y)
    acc = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    return loss, acc

def make_batches(x_train_char_seq, x_train_word, x_train_char, y_train, batch_size, nb_epochs, shuffle=False):
    dataset = None
    if FLAGS["model.emb_mode"] == 1:
        dataset = tf.data.Dataset.from_tensor_slices(((x_train_char_seq,), y_train))
    elif FLAGS["model.emb_mode"] == 2:
        dataset = tf.data.Dataset.from_tensor_slices(((x_train_word,), y_train))
    elif FLAGS["model.emb_mode"] == 3:
        dataset = tf.data.Dataset.from_tensor_slices(((x_train_char_seq, x_train_word), y_train))
    elif FLAGS["model.emb_mode"] == 4:
        dataset = tf.data.Dataset.from_tensor_slices(((x_train_word, x_train_char), y_train))
    elif FLAGS["model.emb_mode"] == 5:
        dataset = tf.data.Dataset.from_tensor_slices(((x_train_char_seq, x_train_word, x_train_char), y_train))
    else:
        raise ValueError("Invalid emb_mode: {}".format(FLAGS["model.emb_mode"]))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)

    dataset = dataset.repeat(nb_epochs).batch(batch_size)
    return dataset

def prep_batches(batch):
    x_batch, y_batch = batch
    x_batch_list = []

    if FLAGS["model.emb_mode"] in [1]:
        x_char_seq = pad_sequences(x_batch[0], maxlen=FLAGS["data.max_len_chars"], padding='post')
        x_batch_list.append(x_char_seq)
    elif FLAGS["model.emb_mode"] in [2]:
        x_word = pad_sequences(x_batch[0], maxlen=FLAGS["data.max_len_words"], padding='post')
        x_batch_list.append(x_word)
    elif FLAGS["model.emb_mode"] in [3]:
        x_char_seq = pad_sequences(x_batch[0], maxlen=FLAGS["data.max_len_chars"], padding='post')
        x_word = pad_sequences(x_batch[1], maxlen=FLAGS["data.max_len_words"], padding='post')
        x_batch_list.extend([x_char_seq, x_word])
    elif FLAGS["model.emb_mode"] in [4]:
        x_word = pad_sequences(x_batch[0], maxlen=FLAGS["data.max_len_words"], padding='post')
        x_char = pad_sequences(x_batch[1], maxlen=FLAGS["data.max_len_words"], padding='post')
        x_char_pad_idx = None  # Adjust as necessary
        x_batch_list.extend([x_word, x_char, x_char_pad_idx])
    elif FLAGS["model.emb_mode"] in [5]:
        x_char_seq = pad_sequences(x_batch[0], maxlen=FLAGS["data.max_len_chars"], padding='post')
        x_word = pad_sequences(x_batch[1], maxlen=FLAGS["data.max_len_words"], padding='post')
        x_char = pad_sequences(x_batch[2], maxlen=FLAGS["data.max_len_words"], padding='post')
        x_char_pad_idx = None  # Adjust as necessary
        x_batch_list.extend([x_char_seq, x_word, x_char, x_char_pad_idx])
    else:
        raise ValueError("Invalid emb_mode: {}".format(FLAGS["model.emb_mode"]))

    y_batch = y_batch.astype(np.float32)
    return x_batch_list, y_batch

# Instantiate the model
cnn = TextCNN(
    char_ngram_vocab_size=len(ngrams_dict) + 1,
    word_ngram_vocab_size=len(words_dict) + 1,
    char_vocab_size=len(chars_dict) + 1,
    embedding_size=FLAGS["model.emb_dim"],
    word_seq_len=FLAGS["data.max_len_words"],
    char_seq_len=FLAGS["data.max_len_chars"],
    l2_reg_lambda=FLAGS["train.l2_reg_lambda"],
    mode=FLAGS["model.emb_mode"],
    filter_sizes=list(map(int, FLAGS["model.filter_sizes"].split(",")))
)

# Define optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS["train.lr"])

# Prepare datasets
train_dataset = make_batches(x_train_char_seq, x_train_word, x_train_char, y_train, FLAGS["train.batch_size"], FLAGS['train.nb_epochs'], shuffle=True)
test_dataset = make_batches(x_test_char_seq, x_test_word, x_test_char, y_test, FLAGS['train.batch_size'], nb_epochs=1, shuffle=False)

# Training loop
min_dev_loss = float('inf')
dev_loss = float('inf')
dev_acc = 0.0

if not os.path.exists(FLAGS["log.output_dir"]):
    os.makedirs(FLAGS["log.output_dir"])

# Save dictionaries
ngrams_dict_dir = os.path.join(FLAGS["log.output_dir"], "subwords_dict.p")
pickle.dump(ngrams_dict, open(ngrams_dict_dir, "wb"))
words_dict_dir = os.path.join(FLAGS["log.output_dir"], "words_dict.p")
pickle.dump(words_dict, open(words_dict_dir, "wb"))
chars_dict_dir = os.path.join(FLAGS["log.output_dir"], "chars_dict.p")
pickle.dump(chars_dict, open(chars_dict_dir, "wb"))

train_log_dir = os.path.join(FLAGS["log.output_dir"], "train_logs.csv")
val_log_dir = os.path.join(FLAGS["log.output_dir"], "val_logs.csv")

with open(train_log_dir, "w") as f:
    f.write("step,time,loss,acc\n")
with open(val_log_dir, "w") as f:
    f.write("step,time,loss,acc\n")

checkpoint_dir = os.path.join(FLAGS["log.output_dir"], "checkpoints")
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
checkpoint_prefix = os.path.join(checkpoint_dir, "model")
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=cnn)

step = 0
total_steps = int(np.ceil(len(y_train) / FLAGS["train.batch_size"]) * FLAGS["train.nb_epochs"])

progress_bar = tqdm(train_dataset, total=total_steps, desc="Training", ncols=100)

for batch in progress_bar:
    x_batch_list, y_batch = prep_batches(batch)
    loss, acc = train_dev_step(cnn, x_batch_list, y_batch, emb_mode=FLAGS["model.emb_mode"], optimizer=optimizer, is_train=True)
    step += 1

    if step % FLAGS["log.print_every"] == 0:
        with open(train_log_dir, "a") as f:
            f.write("{:d},{:s},{:e},{:e}\n".format(step, datetime.datetime.now().isoformat(), loss.numpy(), acc.numpy()))
        progress_bar.set_postfix({'loss': loss.numpy(), 'acc': acc.numpy()})

    if step % FLAGS["log.eval_every"] == 0:
        total_loss = 0.0
        total_acc = 0.0
        nb_batches = 0

        for test_batch in test_dataset:
            x_test_batch_list, y_test_batch = prep_batches(test_batch)
            loss_val, acc_val = train_dev_step(cnn, x_test_batch_list, y_test_batch, emb_mode=FLAGS["model.emb_mode"], optimizer=optimizer, is_train=False)
            batch_size = y_test_batch.shape[0]
            total_loss += loss_val.numpy() * batch_size
            total_acc += acc_val.numpy() * batch_size
            nb_batches += batch_size

        dev_loss = total_loss / nb_batches
        dev_acc = total_acc / nb_batches

        with open(val_log_dir, "a") as f:
            f.write("{:d},{:s},{:e},{:e}\n".format(step, datetime.datetime.now().isoformat(), dev_loss, dev_acc))
        print(f"\nValidation Loss: {dev_loss}, Validation Accuracy: {dev_acc}\n")

    if step % FLAGS["log.checkpoint_every"] == 0:
        if dev_loss < min_dev_loss:
            min_dev_loss = dev_loss
            checkpoint.save(file_prefix=checkpoint_prefix)

# Save the final model
cnn.save(os.path.join(FLAGS["log.output_dir"], "final_model.h5"))

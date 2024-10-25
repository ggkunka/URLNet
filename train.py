import argparse
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from TextCNN import TextCNN
from utils import *
import pickle
import datetime
import os

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

y_train = np.array(y_train)
y_test = np.array(y_test)

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

# Pad sequences
x_train_char_seq_padded = pad_sequences(x_train_char_seq, maxlen=FLAGS["data.max_len_chars"], padding='post')
x_test_char_seq_padded = pad_sequences(x_test_char_seq, maxlen=FLAGS["data.max_len_chars"], padding='post')

# For emb_mode that uses x_word or x_char
x_train_word_padded = pad_sequences(x_train_word, maxlen=FLAGS["data.max_len_words"], padding='post')
x_test_word_padded = pad_sequences(x_test_word, maxlen=FLAGS["data.max_len_words"], padding='post')

x_train_char_padded = pad_sequences(x_train_char, maxlen=FLAGS["data.max_len_words"], padding='post')
x_test_char_padded = pad_sequences(x_test_char, maxlen=FLAGS["data.max_len_words"], padding='post')

# Prepare datasets
def make_batches(x_char_seq, x_word, x_char, y, batch_size, shuffle=True):
    dataset = None
    if FLAGS["model.emb_mode"] == 1:
        dataset = tf.data.Dataset.from_tensor_slices(((x_char_seq,), y))
    elif FLAGS["model.emb_mode"] == 2:
        dataset = tf.data.Dataset.from_tensor_slices(((x_word,), y))
    elif FLAGS["model.emb_mode"] == 3:
        dataset = tf.data.Dataset.from_tensor_slices(((x_char_seq, x_word), y))
    elif FLAGS["model.emb_mode"] == 4:
        dataset = tf.data.Dataset.from_tensor_slices(((x_word, x_char), y))
    elif FLAGS["model.emb_mode"] == 5:
        dataset = tf.data.Dataset.from_tensor_slices(((x_char_seq, x_word, x_char), y))
    else:
        raise ValueError("Invalid emb_mode: {}".format(FLAGS["model.emb_mode"]))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size)
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
        x_batch_list.extend([x_word, x_char])
    elif FLAGS["model.emb_mode"] in [5]:
        x_char_seq = pad_sequences(x_batch[0], maxlen=FLAGS["data.max_len_chars"], padding='post')
        x_word = pad_sequences(x_batch[1], maxlen=FLAGS["data.max_len_words"], padding='post')
        x_char = pad_sequences(x_batch[2], maxlen=FLAGS["data.max_len_words"], padding='post')
        x_batch_list.extend([x_char_seq, x_word, x_char])
    else:
        raise ValueError("Invalid emb_mode: {}".format(FLAGS["model.emb_mode"]))

    y_batch = y_batch.astype(np.float32)
    return x_batch_list, y_batch

# Training loop
#train_dataset = make_batches(x_train_char_seq, x_train_word, x_train_char, y_train, FLAGS["train.batch_size"], shuffle=True)
#test_dataset = make_batches(x_test_char_seq, x_test_word, x_test_char, y_test, FLAGS["train.batch_size"], shuffle=False)
train_dataset = make_batches(
    x_train_char_seq_padded,
    x_train_word_padded,
    x_train_char_padded,
    y_train,
    FLAGS["train.batch_size"],
    shuffle=True
)

test_dataset = make_batches(
    x_test_char_seq_padded,
    x_test_word_padded,
    x_test_char_padded,
    y_test,
    FLAGS["train.batch_size"],
    shuffle=False
)


loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

val_loss = tf.keras.metrics.Mean(name='val_loss')
val_accuracy = tf.keras.metrics.CategoricalAccuracy(name='val_accuracy')

checkpoint_dir = os.path.join(FLAGS["log.output_dir"], "checkpoints")
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=cnn)

if not os.path.exists(FLAGS["log.output_dir"]):
    os.makedirs(FLAGS["log.output_dir"])

# Save dictionaries
with open(os.path.join(FLAGS["log.output_dir"], "subwords_dict.p"), "wb") as f:
    pickle.dump(ngrams_dict, f)
with open(os.path.join(FLAGS["log.output_dir"], "words_dict.p"), "wb") as f:
    pickle.dump(words_dict, f)
with open(os.path.join(FLAGS["log.output_dir"], "chars_dict.p"), "wb") as f:
    pickle.dump(chars_dict, f)

for epoch in range(FLAGS["train.nb_epochs"]):
    print(f"\nStart of epoch {epoch+1}")
    train_loss.reset_states()
    train_accuracy.reset_states()

    for batch in tqdm(train_dataset, desc="Training"):
        x_batch_list, y_batch = prep_batches(batch)
        inputs = {}
        if FLAGS["model.emb_mode"] in [4, 5]:
            inputs['input_x_char'] = x_batch_list[1 if FLAGS["model.emb_mode"] == 4 else 2]
            inputs['input_x_char_pad_idx'] = None  # Adjust if needed
        if FLAGS["model.emb_mode"] in [2, 3, 4, 5]:
            inputs['input_x_word'] = x_batch_list[0 if FLAGS["model.emb_mode"] in [2, 4] else 1]
        if FLAGS["model.emb_mode"] in [1, 3, 5]:
            inputs['input_x_char_seq'] = x_batch_list[0]

        with tf.GradientTape() as tape:
            logits = cnn(inputs, training=True)
            loss = loss_fn(y_batch, logits)

        gradients = tape.gradient(loss, cnn.trainable_variables)
        optimizer.apply_gradients(zip(gradients, cnn.trainable_variables))

        train_loss(loss)
        train_accuracy(y_batch, logits)

    print(f"Epoch {epoch+1}, Loss: {train_loss.result()}, Accuracy: {train_accuracy.result()}")

    # Validation
    val_loss.reset_states()
    val_accuracy.reset_states()

    for batch in test_dataset:
        x_batch_list, y_batch = prep_batches(batch)
        inputs = {}
        if FLAGS["model.emb_mode"] in [4, 5]:
            inputs['input_x_char'] = x_batch_list[1 if FLAGS["model.emb_mode"] == 4 else 2]
            inputs['input_x_char_pad_idx'] = None  # Adjust if needed
        if FLAGS["model.emb_mode"] in [2, 3, 4, 5]:
            inputs['input_x_word'] = x_batch_list[0 if FLAGS["model.emb_mode"] in [2, 4] else 1]
        if FLAGS["model.emb_mode"] in [1, 3, 5]:
            inputs['input_x_char_seq'] = x_batch_list[0]

        logits = cnn(inputs, training=False)
        loss = loss_fn(y_batch, logits)

        val_loss(loss)
        val_accuracy(y_batch, logits)

    print(f"Validation Loss: {val_loss.result()}, Validation Accuracy: {val_accuracy.result()}")

    # Save checkpoint
    checkpoint.save(file_prefix=checkpoint_prefix)

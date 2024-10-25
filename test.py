import argparse
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils import *
import pickle
from TextCNN import TextCNN

parser = argparse.ArgumentParser(description="Test URLNet model")

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
parser.add_argument('--data.data_dir', type=str, default='test_10000.txt', metavar="DATADIR",
                    help="Location of data file")
default_delimit_mode = 1
parser.add_argument("--data.delimit_mode", type=int, default=default_delimit_mode, metavar="DLMODE",
                    help="0: delimit by special chars, 1: delimit by special chars + each char as a word (default: {})".format(default_delimit_mode))
parser.add_argument('--data.subword_dict_dir', type=str, default="runs/10000/subwords_dict.p", metavar="SUBWORD_DICT",
                    help="Directory of the subword dictionary")
parser.add_argument('--data.word_dict_dir', type=str, default="runs/10000/words_dict.p", metavar="WORD_DICT",
                    help="Directory of the word dictionary")
parser.add_argument('--data.char_dict_dir', type=str, default="runs/10000/chars_dict.p", metavar="CHAR_DICT",
                    help="Directory of the character dictionary")

# Model arguments
default_emb_dim = 32
parser.add_argument('--model.emb_dim', type=int, default=default_emb_dim, metavar="EMBDIM",
                    help="Embedding dimension size (default: {})".format(default_emb_dim))
parser.add_argument('--model.filter_sizes', type=str, default="3,4,5,6", metavar="FILTERSIZES",
                    help="Filter sizes of the convolution layer")
default_emb_mode = 1
parser.add_argument('--model.emb_mode', type=int, default=default_emb_mode, metavar="EMBMODE",
                    help="1: charCNN, 2: wordCNN, 3: char + wordCNN, 4: char-level wordCNN, 5: char + char-level wordCNN (default: {})".format(default_emb_mode))

# Test arguments
default_batch_size = 128
parser.add_argument('--test.batch_size', type=int, default=default_batch_size, metavar="BATCHSIZE",
                    help="Size of each test batch (default: {})".format(default_batch_size))

# Log arguments
parser.add_argument('--log.output_dir', type=str, default="runs/10000/", metavar="OUTPUTDIR",
                    help="Directory to save the test results")
parser.add_argument('--log.checkpoint_dir', type=str, default="runs/10000/checkpoints/", metavar="CHECKPOINTDIR",
                    help="Directory of the learned model")

FLAGS = vars(parser.parse_args())
for key, val in FLAGS.items():
    print("{}={}".format(key, val))

# Load data
urls, labels = read_data(FLAGS["data.data_dir"])

x, word_reverse_dict = get_word_vocab(urls, FLAGS["data.max_len_words"])
word_x = get_words(x, word_reverse_dict, FLAGS["data.delimit_mode"], urls)

ngram_dict = pickle.load(open(FLAGS["data.subword_dict_dir"], "rb"))
print("Size of subword vocabulary (train): {}".format(len(ngram_dict)))
word_dict = pickle.load(open(FLAGS["data.word_dict_dir"], "rb"))
print("Size of word vocabulary (train): {}".format(len(word_dict)))
ngrams_dict = ngram_dict
chars_dict = pickle.load(open(FLAGS["data.char_dict_dir"], "rb"))
print("Size of character vocabulary (train): {}".format(len(chars_dict)))

ngramed_id_x, worded_id_x = ngram_id_x_from_dict(word_x, FLAGS["data.max_len_subwords"], ngram_dict, word_dict)
chared_id_x = char_id_x(urls, chars_dict, FLAGS["data.max_len_chars"])

print("Number of testing URLs: {}".format(len(labels)))

# Prepare datasets
def make_batches(x_char_seq, x_word, x_char, batch_size):
    dataset = None
    if FLAGS["model.emb_mode"] == 1:
        dataset = tf.data.Dataset.from_tensor_slices((x_char_seq))
    elif FLAGS["model.emb_mode"] == 2:
        dataset = tf.data.Dataset.from_tensor_slices((x_word))
    elif FLAGS["model.emb_mode"] == 3:
        dataset = tf.data.Dataset.from_tensor_slices((x_char_seq, x_word))
    elif FLAGS["model.emb_mode"] == 4:
        dataset = tf.data.Dataset.from_tensor_slices((x_word, x_char))
    elif FLAGS["model.emb_mode"] == 5:
        dataset = tf.data.Dataset.from_tensor_slices((x_char_seq, x_word, x_char))
    else:
        raise ValueError("Invalid emb_mode: {}".format(FLAGS["model.emb_mode"]))
    dataset = dataset.batch(batch_size)
    return dataset

def prep_batches(batch):
    x_batch_list = []
    if FLAGS["model.emb_mode"] == 1:
        x_char_seq = pad_sequences(batch, maxlen=FLAGS["data.max_len_chars"], padding='post')
        x_batch_list.append(x_char_seq)
    elif FLAGS["model.emb_mode"] == 2:
        x_word = pad_sequences(batch, maxlen=FLAGS["data.max_len_words"], padding='post')
        x_batch_list.append(x_word)
    elif FLAGS["model.emb_mode"] == 3:
        x_char_seq = pad_sequences(batch[0], maxlen=FLAGS["data.max_len_chars"], padding='post')
        x_word = pad_sequences(batch[1], maxlen=FLAGS["data.max_len_words"], padding='post')
        x_batch_list.extend([x_char_seq, x_word])
    elif FLAGS["model.emb_mode"] == 4:
        x_word = pad_sequences(batch[0], maxlen=FLAGS["data.max_len_words"], padding='post')
        x_char = pad_sequences(batch[1], maxlen=FLAGS["data.max_len_words"], padding='post')
        x_batch_list.extend([x_word, x_char])
    elif FLAGS["model.emb_mode"] == 5:
        x_char_seq = pad_sequences(batch[0], maxlen=FLAGS["data.max_len_chars"], padding='post')
        x_word = pad_sequences(batch[1], maxlen=FLAGS["data.max_len_words"], padding='post')
        x_char = pad_sequences(batch[2], maxlen=FLAGS["data.max_len_words"], padding='post')
        x_batch_list.extend([x_char_seq, x_word, x_char])
    else:
        raise ValueError("Invalid emb_mode: {}".format(FLAGS["model.emb_mode"]))
    return x_batch_list

# Load the model
cnn = TextCNN(
    char_ngram_vocab_size=len(ngrams_dict) + 1,
    word_ngram_vocab_size=len(word_dict) + 1,
    char_vocab_size=len(chars_dict) + 1,
    embedding_size=FLAGS["model.emb_dim"],
    word_seq_len=FLAGS["data.max_len_words"],
    char_seq_len=FLAGS["data.max_len_chars"],
    mode=FLAGS["model.emb_mode"],
    filter_sizes=list(map(int, FLAGS["model.filter_sizes"].split(",")))
)

# Restore the checkpoint
checkpoint = tf.train.Checkpoint(model=cnn)
latest_checkpoint = tf.train.latest_checkpoint(FLAGS["log.checkpoint_dir"])
if latest_checkpoint:
    checkpoint.restore(latest_checkpoint)
    print("Restored from {}".format(latest_checkpoint))
else:
    print("No checkpoint found at {}".format(FLAGS["log.checkpoint_dir"]))

# Prepare dataset
test_dataset = make_batches(chared_id_x, worded_id_x, chared_id_x, FLAGS["test.batch_size"])

# Evaluation
all_predictions = []
all_scores = []

for batch in tqdm(test_dataset, desc="Testing"):
    x_batch_list = prep_batches(batch)
    inputs = {}
    if FLAGS["model.emb_mode"] in [4, 5]:
        inputs['input_x_char'] = x_batch_list[1 if FLAGS["model.emb_mode"] == 4 else 2]
        inputs['input_x_char_pad_idx'] = None  # Adjust if needed
    if FLAGS["model.emb_mode"] in [2, 3, 4, 5]:
        inputs['input_x_word'] = x_batch_list[0 if FLAGS["model.emb_mode"] in [2, 4] else 1]
    if FLAGS["model.emb_mode"] in [1, 3, 5]:
        inputs['input_x_char_seq'] = x_batch_list[0]

    logits = cnn(inputs, training=False)
    preds = tf.argmax(logits, axis=1).numpy()
    scores = logits.numpy()
    all_predictions.extend(preds)
    all_scores.extend(scores)

# Save test results
save_test_result(labels, all_predictions, all_scores, FLAGS["log.output_dir"] + "test_results.txt")

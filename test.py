import argparse
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils import *
import pickle
from TextCNN import TextCNN

parser = argparse.ArgumentParser(description="Test URLNet model")

# [No changes in the argument parser]

FLAGS = vars(parser.parse_args())
for key, val in FLAGS.items():
    print("{}={}".format(key, val))

# Load data
urls, labels = read_data(FLAGS["data.data_dir"])

# **Change 1: Map labels from -1 and 1 to 0 and 1**
labels = [0 if label == -1 else 1 for label in labels]

x, word_reverse_dict = get_word_vocab(urls, FLAGS["data.max_len_words"])
word_x = get_words(x, word_reverse_dict, FLAGS["data.delimit_mode"], urls)

# Load dictionaries
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

# Pad sequences based on the embedding mode
if FLAGS["model.emb_mode"] == 1:
    x_char_seq_padded = pad_sequences(chared_id_x, maxlen=FLAGS["data.max_len_chars"], padding='post')
elif FLAGS["model.emb_mode"] == 2:
    x_word_padded = pad_sequences(worded_id_x, maxlen=FLAGS["data.max_len_words"], padding='post')
elif FLAGS["model.emb_mode"] == 3:
    x_char_seq_padded = pad_sequences(chared_id_x, maxlen=FLAGS["data.max_len_chars"], padding='post')
    x_word_padded = pad_sequences(worded_id_x, maxlen=FLAGS["data.max_len_words"], padding='post')
elif FLAGS["model.emb_mode"] == 4:
    x_word_padded = pad_sequences(worded_id_x, maxlen=FLAGS["data.max_len_words"], padding='post')
    x_char_padded = pad_sequences(chared_id_x, maxlen=FLAGS["data.max_len_words"], padding='post')
elif FLAGS["model.emb_mode"] == 5:
    x_char_seq_padded = pad_sequences(chared_id_x, maxlen=FLAGS["data.max_len_chars"], padding='post')
    x_word_padded = pad_sequences(worded_id_x, maxlen=FLAGS["data.max_len_words"], padding='post')
    x_char_padded = pad_sequences(chared_id_x, maxlen=FLAGS["data.max_len_words"], padding='post')
else:
    raise ValueError("Invalid emb_mode: {}".format(FLAGS["model.emb_mode"]))

# Prepare datasets
def make_batches(batch_size):
    if FLAGS["model.emb_mode"] == 1:
        dataset = tf.data.Dataset.from_tensor_slices((x_char_seq_padded, labels))
    elif FLAGS["model.emb_mode"] == 2:
        dataset = tf.data.Dataset.from_tensor_slices((x_word_padded, labels))
    elif FLAGS["model.emb_mode"] == 3:
        dataset = tf.data.Dataset.from_tensor_slices(((x_char_seq_padded, x_word_padded), labels))
    elif FLAGS["model.emb_mode"] == 4:
        dataset = tf.data.Dataset.from_tensor_slices(((x_word_padded, x_char_padded), labels))
    elif FLAGS["model.emb_mode"] == 5:
        dataset = tf.data.Dataset.from_tensor_slices(((x_char_seq_padded, x_word_padded, x_char_padded), labels))
    else:
        raise ValueError("Invalid emb_mode: {}".format(FLAGS["model.emb_mode"]))
    dataset = dataset.batch(batch_size)
    return dataset

def prep_batches(batch):
    x_batch = batch[0]
    y_batch = batch[1]
    x_batch_list = []
    if FLAGS["model.emb_mode"] == 1:
        x_batch_list.append(x_batch)
    elif FLAGS["model.emb_mode"] == 2:
        x_batch_list.append(x_batch)
    elif FLAGS["model.emb_mode"] == 3:
        x_batch_list.extend([x_batch[0], x_batch[1]])
    elif FLAGS["model.emb_mode"] == 4:
        x_batch_list.extend([x_batch[0], x_batch[1]])
    elif FLAGS["model.emb_mode"] == 5:
        x_batch_list.extend([x_batch[0], x_batch[1], x_batch[2]])
    else:
        raise ValueError("Invalid emb_mode: {}".format(FLAGS["model.emb_mode"]))
    y_batch = tf.cast(y_batch, tf.float32)
    y_batch = tf.reshape(y_batch, (-1, 1))  # Ensure labels are of shape (batch_size, 1)
    return x_batch_list, y_batch

# Load the model
cnn = TextCNN(
    char_ngram_vocab_size=len(ngrams_dict) + 1,
    word_ngram_vocab_size=len(word_dict) + 1,
    char_vocab_size=len(chars_dict) + 1,
    embedding_size=FLAGS["model.emb_dim"],
    word_seq_len=FLAGS["data.max_len_words"],
    char_seq_len=FLAGS["data.max_len_chars"],
    mode=FLAGS["model.emb_mode"],
    filter_sizes=list(map(int, FLAGS["model.filter_sizes"].split(","))),
    num_classes=1,  # Ensure num_classes is set to 1
)

# Restore the checkpoint
checkpoint = tf.train.Checkpoint(model=cnn)
latest_checkpoint = tf.train.latest_checkpoint(FLAGS["log.checkpoint_dir"])
if latest_checkpoint:
    # Use expect_partial() to suppress warnings about optimizer variables
    checkpoint.restore(latest_checkpoint).expect_partial()
    print("Restored from {}".format(latest_checkpoint))
else:
    print("No checkpoint found at {}".format(FLAGS["log.checkpoint_dir"]))

# Prepare dataset
test_dataset = make_batches(FLAGS["test.batch_size"])

# Evaluation
all_predictions = []
all_scores = []

for batch in tqdm(test_dataset, desc="Testing"):
    x_batch_list, y_batch = prep_batches(batch)
    inputs = {}
    if FLAGS["model.emb_mode"] in [4, 5]:
        inputs['input_x_char'] = x_batch_list[1 if FLAGS["model.emb_mode"] == 4 else 2]
    if FLAGS["model.emb_mode"] in [2, 3, 4, 5]:
        inputs['input_x_word'] = x_batch_list[0 if FLAGS["model.emb_mode"] in [2, 4] else 1]
    if FLAGS["model.emb_mode"] in [1, 3, 5]:
        inputs['input_x_char_seq'] = x_batch_list[0]

    logits = cnn(inputs, training=False)  # Output is probabilities between 0 and 1
    predictions = (logits.numpy() > 0.5).astype(int).flatten()
    all_predictions.extend(predictions)
    all_scores.extend(logits.numpy().flatten())

# Save test results
save_test_result(labels, all_predictions, all_scores, FLAGS["log.output_dir"] + "test_results.txt")

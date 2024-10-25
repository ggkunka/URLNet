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

# [No changes in the argument parser]

FLAGS = vars(parser.parse_args())

for key, val in FLAGS.items():
    print("{}={}".format(key, val))

# Data preparation
urls, labels = read_data(FLAGS["data.data_dir"])

# **Change 1: Map labels from -1 and 1 to 0 and 1**
labels = [0 if label == -1 else 1 for label in labels]

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

# **No changes in building pos_x and neg_x lists, but now labels are 0 and 1**
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

# **Change 2: Use sklearn's train_test_split with stratification**
from sklearn.model_selection import train_test_split

indices = np.arange(len(labels))
x_train_indices, x_test_indices, y_train, y_test = train_test_split(
    indices, labels, test_size=FLAGS["data.dev_pct"], random_state=42, stratify=labels)

# Get corresponding data
x_train_char_seq = [chared_id_x[i] for i in x_train_indices]
x_test_char_seq = [chared_id_x[i] for i in x_test_indices]

# Convert labels to numpy arrays
y_train = np.array(y_train)
y_test = np.array(y_test)

# **Change 3: Instantiate the model with sigmoid activation in the output layer**
cnn = TextCNN(
    char_ngram_vocab_size=len(ngrams_dict) + 1,
    word_ngram_vocab_size=len(words_dict) + 1,
    char_vocab_size=len(chars_dict) + 1,
    embedding_size=FLAGS["model.emb_dim"],
    word_seq_len=FLAGS["data.max_len_words"],
    char_seq_len=FLAGS["data.max_len_chars"],
    l2_reg_lambda=FLAGS["train.l2_reg_lambda"],
    mode=FLAGS["model.emb_mode"],
    filter_sizes=list(map(int, FLAGS["model.filter_sizes"].split(","))),
    num_classes=1,  # **Specify number of classes as 1 for binary classification**
)

# **Change 4: Define optimizer and use BinaryCrossentropy loss function**
optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS["train.lr"])
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)

# Pad sequences for x_char_seq
x_train_char_seq_padded = pad_sequences(x_train_char_seq, maxlen=FLAGS["data.max_len_chars"], padding='post')
x_test_char_seq_padded = pad_sequences(x_test_char_seq, maxlen=FLAGS["data.max_len_chars"], padding='post')

# Prepare datasets
def make_batches(x_char_seq, y, batch_size, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices((x_char_seq, y))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size)
    return dataset

def prep_batches(batch):
    x_batch, y_batch = batch
    x_batch_list = [x_batch]
    y_batch = tf.cast(y_batch, tf.float32)
    y_batch = tf.reshape(y_batch, (-1, 1))  # **Ensure labels are of shape (batch_size, 1)**
    return x_batch_list, y_batch

# Training loop
train_dataset = make_batches(x_train_char_seq_padded, y_train, FLAGS["train.batch_size"], shuffle=True)
test_dataset = make_batches(x_test_char_seq_padded, y_test, FLAGS["train.batch_size"], shuffle=False)

# **Change 5: Use BinaryAccuracy metric**
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')

val_loss = tf.keras.metrics.Mean(name='val_loss')
val_accuracy = tf.keras.metrics.BinaryAccuracy(name='val_accuracy')

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
        inputs = {'input_x_char_seq': x_batch_list[0]}  # Input for emb_mode=1

        with tf.GradientTape() as tape:
            logits = cnn(inputs, training=True)  # **Output will be probabilities between 0 and 1**
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
        inputs = {'input_x_char_seq': x_batch_list[0]}  # Input for emb_mode=1

        logits = cnn(inputs, training=False)
        loss = loss_fn(y_batch, logits)

        val_loss(loss)
        val_accuracy(y_batch, logits)

    print(f"Validation Loss: {val_loss.result()}, Validation Accuracy: {val_accuracy.result()}")

    # Save checkpoint
    checkpoint.save(file_prefix=checkpoint_prefix)

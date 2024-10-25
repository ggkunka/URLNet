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
from sklearn.model_selection import train_test_split

def main():
    parser = argparse.ArgumentParser(description="Train URLNet model")

    # Data arguments
    parser.add_argument('--data_data_dir', type=str, default='data/train_converted.txt', help="Location of data file")
    parser.add_argument('--data_max_len_words', type=int, default=200, help="Maximum length of URL in words")
    parser.add_argument('--data_max_len_chars', type=int, default=200, help="Maximum length of URL in characters")
    parser.add_argument('--data_max_len_subwords', type=int, default=20, help="Maximum length of word in subwords/characters")
    parser.add_argument('--data_min_word_freq', type=int, default=1, help="Minimum frequency of word to build vocabulary")
    parser.add_argument('--data_dev_pct', type=float, default=0.1, help="Percentage of data used for validation")
    parser.add_argument('--data_delimit_mode', type=int, default=0, help="0: special chars, 1: special chars + each char as word")

    # Model arguments
    parser.add_argument('--model_emb_dim', type=int, default=32, help="Embedding dimension size")
    parser.add_argument('--model_filter_sizes', type=str, default="3,4,5,6", help="Filter sizes of the convolution layer")
    parser.add_argument('--model_emb_mode', type=int, default=1, help="1: charCNN, 2: wordCNN, etc.")

    # Training arguments
    parser.add_argument('--train_nb_epochs', type=int, default=5, help="Number of training epochs")
    parser.add_argument('--train_batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('--train_l2_reg_lambda', type=float, default=0.0, help="L2 regularization lambda")
    parser.add_argument('--train_lr', type=float, default=0.001, help="Learning rate")

    # Logging arguments
    parser.add_argument('--log_output_dir', type=str, default="runs/1000_emb1_dlm0_run/", help="Output directory")
    parser.add_argument('--log_print_every', type=int, default=5, help="Print training info every n steps")
    parser.add_argument('--log_eval_every', type=int, default=10, help="Evaluate model every n steps")
    parser.add_argument('--log_checkpoint_every', type=int, default=10, help="Save model every n steps")

    args = parser.parse_args()

    for key, val in vars(args).items():
        print(f"{key}={val}")

    # Data preparation
    urls, labels = read_data(args.data_data_dir)

    # **No need to map labels; they are already 0 and 1**
    # labels = [0 if label == -1 else 1 for label in labels]

    high_freq_words = None
    if args.data_min_word_freq > 1:
        x1, word_reverse_dict = get_word_vocab(urls, args.data_max_len_words, args.data_min_word_freq)
        high_freq_words = sorted(list(word_reverse_dict.values()))
        print(f"Number of words with freq >= {args.data_min_word_freq}: {len(high_freq_words)}")

    x, word_reverse_dict = get_word_vocab(urls, args.data_max_len_words)
    word_x = get_words(x, word_reverse_dict, args.data_delimit_mode, urls)
    ngramed_id_x, ngrams_dict, worded_id_x, words_dict = ngram_id_x(word_x, args.data_max_len_subwords, high_freq_words)

    chars_dict = ngrams_dict
    chared_id_x = char_id_x(urls, chars_dict, args.data_max_len_chars)

    print(f"Size of ngrams_dict: {len(ngrams_dict)}")
    print(f"Size of words_dict: {len(words_dict)}")
    print(f"Size of chars_dict: {len(chars_dict)}")

    # Split data into training and validation sets using stratification
    indices = np.arange(len(labels))
    x_train_indices, x_val_indices, y_train, y_val = train_test_split(
        indices, labels, test_size=args.data_dev_pct, random_state=42, stratify=labels)

    # Get corresponding data
    x_train_char_seq = [chared_id_x[i] for i in x_train_indices]
    x_val_char_seq = [chared_id_x[i] for i in x_val_indices]

    # Pad sequences
    x_train_char_seq_padded = pad_sequences(x_train_char_seq, maxlen=args.data_max_len_chars, padding='post')
    x_val_char_seq_padded = pad_sequences(x_val_char_seq, maxlen=args.data_max_len_chars, padding='post')

    y_train = np.array(y_train)
    y_val = np.array(y_val)

    # Instantiate the model
    cnn = TextCNN(
        char_ngram_vocab_size=len(ngrams_dict) + 1,
        word_ngram_vocab_size=len(words_dict) + 1,
        char_vocab_size=len(chars_dict) + 1,
        embedding_size=args.model_emb_dim,
        word_seq_len=args.data_max_len_words,
        char_seq_len=args.data_max_len_chars,
        l2_reg_lambda=args.train_l2_reg_lambda,
        mode=args.model_emb_mode,
        filter_sizes=list(map(int, args.model_filter_sizes.split(","))),
        num_classes=1,  # Binary classification
    )

    # Define optimizer and loss function
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.train_lr)
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)

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
        y_batch = tf.reshape(y_batch, (-1, 1))
        return x_batch_list, y_batch

    train_dataset = make_batches(x_train_char_seq_padded, y_train, args.train_batch_size, shuffle=True)
    val_dataset = make_batches(x_val_char_seq_padded, y_val, args.train_batch_size, shuffle=False)

    # Define metrics
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')

    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_accuracy = tf.keras.metrics.BinaryAccuracy(name='val_accuracy')

    # Checkpoint setup
    checkpoint_dir = os.path.join(args.log_output_dir, "checkpoints")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=cnn)

    # Save dictionaries
    if not os.path.exists(args.log_output_dir):
        os.makedirs(args.log_output_dir)
    with open(os.path.join(args.log_output_dir, "subwords_dict.p"), "wb") as f:
        pickle.dump(ngrams_dict, f)
    with open(os.path.join(args.log_output_dir, "words_dict.p"), "wb") as f:
        pickle.dump(words_dict, f)
    with open(os.path.join(args.log_output_dir, "chars_dict.p"), "wb") as f:
        pickle.dump(chars_dict, f)

    # Training loop
    for epoch in range(args.train_nb_epochs):
        print(f"\nStart of epoch {epoch+1}")
        train_loss.reset_states()
        train_accuracy.reset_states()

        for batch in tqdm(train_dataset, desc="Training"):
            x_batch_list, y_batch = prep_batches(batch)
            inputs = {'input_x_char_seq': x_batch_list[0]}  # For emb_mode=1

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

        for batch in val_dataset:
            x_batch_list, y_batch = prep_batches(batch)
            inputs = {'input_x_char_seq': x_batch_list[0]}  # For emb_mode=1

            logits = cnn(inputs, training=False)
            loss = loss_fn(y_batch, logits)

            val_loss(loss)
            val_accuracy(y_batch, logits)

        print(f"Validation Loss: {val_loss.result()}, Validation Accuracy: {val_accuracy.result()}")

        # Save checkpoint
        checkpoint.save(file_prefix=checkpoint_prefix)

if __name__ == "__main__":
    main()

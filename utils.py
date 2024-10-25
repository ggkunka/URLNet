import time
import numpy as np
from collections import defaultdict
from bisect import bisect_left
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def read_data(file_dir):
    with open(file_dir) as file:
        urls = []
        labels = []
        for line in file.readlines():
            items = line.strip().split('\t')
            label = int(items[0])
            labels.append(1 if label == 1 else 0)
            url = items[1]
            urls.append(url)
    return urls, labels

def split_url(line, part):
    # Your existing implementation
    pass  # Omitted for brevity since it remains the same

def get_word_vocab(urls, max_length_words, min_word_freq=0):
    tokenizer = Tokenizer(filters='', lower=False, oov_token='<UNKNOWN>')
    tokenizer.fit_on_texts(urls)
    word_counts = tokenizer.word_counts
    if min_word_freq > 1:
        low_freq_words = [word for word, count in word_counts.items() if count < min_word_freq]
        for word in low_freq_words:
            del tokenizer.word_index[word]
            del tokenizer.word_docs[word]
            del tokenizer.word_counts[word]
    x = tokenizer.texts_to_sequences(urls)
    x = pad_sequences(x, maxlen=max_length_words, padding='post', truncating='post')
    reverse_dict = {idx: word for word, idx in tokenizer.word_index.items()}
    return x, reverse_dict

def get_words(x, reverse_dict, delimit_mode, urls=None):
    # Your existing implementation adjusted for updated reverse_dict
    pass  # Omitted for brevity since it remains the same

def get_char_ngrams(ngram_len, word):
    # Your existing implementation
    pass  # Omitted for brevity since it remains the same

def char_id_x(urls, char_dict, max_len_chars):
    chared_id_x = []
    for url in urls:
        url_chars = list(url)[:max_len_chars]
        url_in_char_id = [char_dict.get(c, 0) for c in url_chars]
        chared_id_x.append(url_in_char_id)
    return chared_id_x

def ngram_id_x(word_x, max_len_subwords, high_freq_words=None):
    # Your existing implementation adjusted to use the updated tokenizer
    pass  # Omitted for brevity since it remains the same

def ngram_id_x_from_dict(word_x, max_len_subwords, ngram_dict, word_dict=None):
    # Your existing implementation adjusted for updated dicts
    pass  # Omitted for brevity since it remains the same

def is_in(a, x):
    i = bisect_left(a, x)
    return i != len(a) and a[i] == x

def prep_train_test(pos_x, neg_x, dev_pct):
    np.random.seed(10)
    pos_x_shuffled = np.random.permutation(pos_x)
    dev_idx = -1 * int(dev_pct * float(len(pos_x)))
    pos_train = pos_x_shuffled[:dev_idx]
    pos_test = pos_x_shuffled[dev_idx:]

    neg_x_shuffled = np.random.permutation(neg_x)
    dev_idx = -1 * int(dev_pct * float(len(neg_x)))
    neg_train = neg_x_shuffled[:dev_idx]
    neg_test = neg_x_shuffled[dev_idx:]

    x_train = np.concatenate([pos_train, neg_train])
    y_train = np.array([1]*len(pos_train) + [0]*len(neg_train))
    x_test = np.concatenate([pos_test, neg_test])
    y_test = np.array([1]*len(pos_test) + [0]*len(neg_test))

    y_train = to_categorical(y_train, num_classes=2)
    y_test = to_categorical(y_test, num_classes=2)

    shuffle_indices = np.random.permutation(len(x_train))
    x_train = x_train[shuffle_indices]
    y_train = y_train[shuffle_indices]

    shuffle_indices = np.random.permutation(len(x_test))
    x_test = x_test[shuffle_indices]
    y_test = y_test[shuffle_indices]

    print("Train Mal/Ben split: {}/{}".format(len(pos_train), len(neg_train)))
    print("Test Mal/Ben split: {}/{}".format(len(pos_test), len(neg_test)))
    print("Train/Test split: {}/{}".format(len(y_train), len(y_test)))
    print("Train/Test split: {}/{}".format(len(x_train), len(x_test)))

    return x_train, y_train, x_test, y_test

def get_ngramed_id_x(x_idxs, ngramed_id_x):
    return [ngramed_id_x[idx] for idx in x_idxs]

def pad_seq(urls, max_d1=0, max_d2=0, embedding_size=128):
    # Adjusted implementation
    pass  # Omitted for brevity since it may not be used in the updated code

def pad_seq_in_word(urls, max_d1=0):
    # Adjusted implementation
    pass  # Omitted for brevity since we're using Keras pad_sequences

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    # This function may not be necessary as we're using tf.data.Dataset
    pass

def save_test_result(labels, all_predictions, all_scores, output_dir):
    output_labels = [label if label == 1 else -1 for label in labels]
    output_preds = [pred if pred == 1 else -1 for pred in all_predictions]
    softmax_scores = [softmax(score) for score in all_scores]
    with open(output_dir, "w") as file:
        file.write("label\tpredict\tscore\n")
        for label, pred, score in zip(output_labels, output_preds, softmax_scores):
            file.write(f"{label}\t{pred}\t{score[1]}\n")

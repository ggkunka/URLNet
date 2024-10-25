import time
import os
import numpy as np
from collections import defaultdict
from bisect import bisect_left
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

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
    if line.startswith("http://"):
        line = line[7:]
    if line.startswith("https://"):
        line = line[8:]
    if line.startswith("ftp://"):
        line = line[6:]
    if line.startswith("www."):
        line = line[4:]
    slash_pos = line.find('/')
    if slash_pos > 0 and slash_pos < len(line)-1:
        primarydomain = line[:slash_pos]
        path_argument = line[slash_pos+1:]
        path_argument_tokens = path_argument.split('/')
        pathtoken = "/".join(path_argument_tokens[:-1])
        last_pathtoken = path_argument_tokens[-1]
        if len(path_argument_tokens) > 2 and last_pathtoken == '':
            pathtoken = "/".join(path_argument_tokens[:-2])
            last_pathtoken = path_argument_tokens[-2]
        question_pos = last_pathtoken.find('?')
        if question_pos != -1:
            argument = last_pathtoken[question_pos+1:]
            pathtoken = pathtoken + "/" + last_pathtoken[:question_pos]
        else:
            argument = ""
            pathtoken = pathtoken + "/" + last_pathtoken
        last_slash_pos = pathtoken.rfind('/')
        sub_dir = pathtoken[:last_slash_pos]
        filename = pathtoken[last_slash_pos+1:]
        file_last_dot_pos = filename.rfind('.')
        if file_last_dot_pos != -1:
            file_extension = filename[file_last_dot_pos+1:]
            filename = filename[:file_last_dot_pos]
        else:
            file_extension = ""
    elif slash_pos == 0:
        primarydomain = line[1:]
        pathtoken = ""
        argument = ""
        sub_dir = ""
        filename = ""
        file_extension = ""
    elif slash_pos == len(line)-1:
        primarydomain = line[:-1]
        pathtoken = ""
        argument = ""
        sub_dir = ""
        filename = ""
        file_extension = ""
    else:
        primarydomain = line
        pathtoken = ""
        argument = ""
        sub_dir = ""
        filename = ""
        file_extension = ""
    if part == 'pd':
        return primarydomain
    elif part == 'path':
        return pathtoken
    elif part == 'argument':
        return argument
    elif part == 'sub_dir':
        return sub_dir
    elif part == 'filename':
        return filename
    elif part == 'fe':
        return file_extension
    elif part == 'others':
        return pathtoken + '?' + argument if argument else pathtoken
    else:
        return primarydomain, pathtoken, argument, sub_dir, filename, file_extension

def get_word_vocab(urls, max_length_words, min_word_freq=0):
    tokenizer = Tokenizer(filters='', lower=False, oov_token='<UNK>')
    start = time.time()
    tokenizer.fit_on_texts(urls)
    if min_word_freq > 1:
        word_counts = tokenizer.word_counts.copy()
        low_freq_words = [word for word, count in word_counts.items() if count < min_word_freq]
        for word in low_freq_words:
            del tokenizer.word_index[word]
            del tokenizer.word_counts[word]
        tokenizer.word_index = {word: idx+1 for idx, word in enumerate(tokenizer.word_index.keys())}
        tokenizer.index_word = {idx: word for word, idx in tokenizer.word_index.items()}
    x = tokenizer.texts_to_sequences(urls)
    x = pad_sequences(x, maxlen=max_length_words, padding='post', truncating='post')
    print("Finished building vocabulary and mapping to x in {:.2f} seconds".format(time.time() - start))
    vocab_dict = tokenizer.word_index
    reverse_dict = tokenizer.index_word
    print("Size of word vocabulary: {}".format(len(reverse_dict)))
    return x, reverse_dict

def get_words(x, reverse_dict, delimit_mode, urls=None):
    processed_x = []
    if delimit_mode == 0:
        for url in x:
            words = [reverse_dict[word_id] for word_id in url if word_id != 0]
            processed_x.append(words)
    elif delimit_mode == 1:
        for i in range(len(x)):
            word_url = x[i]
            raw_url = urls[i]
            words = []
            idx_in_raw = 0
            for word_id in word_url:
                if word_id == 0:
                    words.extend(list(raw_url[idx_in_raw:]))
                    break
                word = reverse_dict[word_id]
                idx = raw_url.find(word, idx_in_raw)
                if idx_in_raw < idx:
                    special_chars = list(raw_url[idx_in_raw:idx])
                    words.extend(special_chars)
                words.append(word)
                idx_in_raw = idx + len(word)
            else:
                words.extend(list(raw_url[idx_in_raw:]))
            processed_x.append(words)
    return processed_x

def get_char_ngrams(ngram_len, word):
    word = "<" + word + ">"
    chars = list(word)
    ngrams = ["".join(chars[i:i+ngram_len]) for i in range(len(chars)-ngram_len+1)]
    return ngrams

def char_id_x(urls, char_dict, max_len_chars):
    chared_id_x = []
    for url in urls:
        url_chars = list(url)[:max_len_chars]
        url_in_char_id = [char_dict.get(c, 0) for c in url_chars]
        chared_id_x.append(url_in_char_id)
    return chared_id_x

def ngram_id_x(word_x, max_len_subwords, high_freq_words=None):
    char_ngram_len = 1
    all_ngrams = set()
    ngramed_x = []
    all_words = set()
    worded_x = []
    for url in word_x:
        url_in_ngrams = []
        url_in_words = []
        words = url
        for word in words:
            ngrams = get_char_ngrams(char_ngram_len, word)
            if (len(ngrams) > max_len_subwords) or \
               (high_freq_words is not None and len(word) > 1 and not is_in(high_freq_words, word)):
                all_ngrams.update(ngrams[:max_len_subwords])
                url_in_ngrams.append(ngrams[:max_len_subwords])
                all_words.add("<UNKNOWN>")
                url_in_words.append("<UNKNOWN>")
            else:
                all_ngrams.update(ngrams)
                url_in_ngrams.append(ngrams)
                all_words.add(word)
                url_in_words.append(word)
        ngramed_x.append(url_in_ngrams)
        worded_x.append(url_in_words)
    ngrams_dict = {ngram: idx+1 for idx, ngram in enumerate(all_ngrams)}
    print("Size of ngram vocabulary: {}".format(len(ngrams_dict)))
    words_dict = {word: idx+1 for idx, word in enumerate(all_words)}
    print("Size of word vocabulary: {}".format(len(words_dict)))
    print("Index of <UNKNOWN> word: {}".format(words_dict["<UNKNOWN>"]))
    ngramed_id_x = [[[ngrams_dict.get(ngram, 0) for ngram in ngramed_word] for ngramed_word in ngramed_url] for ngramed_url in ngramed_x]
    worded_id_x = [[words_dict.get(word, 0) for word in worded_url] for worded_url in worded_x]
    return ngramed_id_x, ngrams_dict, worded_id_x, words_dict

def ngram_id_x_from_dict(word_x, max_len_subwords, ngram_dict, word_dict=None):
    char_ngram_len = 1
    print("Index of <UNKNOWN> word: {}".format(word_dict["<UNKNOWN>"]))
    ngramed_id_x = []
    worded_id_x = []
    word_vocab = sorted(word_dict.keys()) if word_dict else []
    for url in word_x:
        url_in_ngrams = []
        url_in_words = []
        words = url
        for word in words:
            ngrams = get_char_ngrams(char_ngram_len, word)
            if len(ngrams) > max_len_subwords:
                word = "<UNKNOWN>"
            ngrams_id = [ngram_dict.get(ngram, 0) for ngram in ngrams]
            url_in_ngrams.append(ngrams_id)
            word_id = word_dict.get(word, word_dict["<UNKNOWN>"])
            url_in_words.append(word_id)
        ngramed_id_x.append(url_in_ngrams)
        worded_id_x.append(url_in_words)
    return ngramed_id_x, worded_id_x

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
    if max_d1 == 0 and max_d2 == 0:
        max_d1 = max(len(url) for url in urls)
        max_d2 = max(len(word) for url in urls for word in url)
    pad_idx = np.zeros((len(urls), max_d1, max_d2, embedding_size))
    pad_urls = np.zeros((len(urls), max_d1, max_d2))
    pad_vec = np.ones(embedding_size)
    for d0, url in enumerate(urls):
        for d1, word in enumerate(url):
            if d1 < max_d1:
                for d2, val in enumerate(word):
                    if d2 < max_d2:
                        pad_urls[d0, d1, d2] = val
                        pad_idx[d0, d1, d2] = pad_vec
    return pad_urls, pad_idx

def pad_seq_in_word(urls, max_d1=0):
    if max_d1 == 0:
        max_d1 = max(len(url) for url in urls)
    pad_urls = np.zeros((len(urls), max_d1))
    for d0, url in enumerate(urls):
        for d1, val in enumerate(url):
            if d1 < max_d1:
                pad_urls[d0, d1] = val
    return pad_urls

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(np.ceil(data_size / batch_size))
    for epoch in range(num_epochs):
        if shuffle:
            data = np.random.permutation(data)
        for batch_num in range(num_batches_per_epoch):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, data_size)
            yield data[start_idx:end_idx]

def save_test_result(labels, all_predictions, all_scores, output_dir):
    output_labels = [label if label == 1 else -1 for label in labels]
    output_preds = [pred if pred == 1 else -1 for pred in all_predictions]
    softmax_scores = [softmax(score) for score in all_scores]
    with open(output_dir, "w") as file:
        file.write("label\tpredict\tscore\n")
        for label, pred, score in zip(output_labels, output_preds, softmax_scores):
            file.write(f"{label}\t{pred}\t{score[1]:.6f}\n")

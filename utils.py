import time
import numpy as np
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
    # Omitted here as it remains the same
    pass

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
    processed_x = []
    if delimit_mode == 0:
        for url in x:
            words = []
            for word_id in url:
                if word_id != 0:
                    words.append(reverse_dict[word_id])
                else:
                    break
            processed_x.append(words)
    elif delimit_mode == 1:
        for i in range(len(x)):
            word_url = x[i]
            raw_url = urls[i]
            words = []
            idx = 0
            for word_id in word_url:
                if word_id == 0:
                    words.extend(list(raw_url[idx:]))
                    break
                else:
                    word = reverse_dict[word_id]
                    word_start_idx = raw_url.find(word, idx)
                    if word_start_idx == -1:
                        words.extend(list(raw_url[idx:]))
                        break
                    special_chars = list(raw_url[idx:word_start_idx])
                    words.extend(special_chars)
                    words.append(word)
                    idx = word_start_idx + len(word)
                    if idx >= len(raw_url):
                        break
            processed_x.append(words)
    return processed_x

def get_char_ngrams(ngram_len, word):
    word = "<" + word + ">"
    chars = list(word)
    ngrams = []
    for i in range(len(chars) - ngram_len + 1):
        ngram = ''.join(chars[i:i+ngram_len])
        ngrams.append(ngram)
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
    high_freq_words_set = set(high_freq_words) if high_freq_words else None

    for url in word_x:
        url_in_ngrams = []
        url_in_words = []
        for word in url:
            ngrams = get_char_ngrams(char_ngram_len, word)
            if (len(ngrams) > max_len_subwords) or \
               (high_freq_words_set and len(word) > 1 and word not in high_freq_words_set):
                ngrams = ngrams[:max_len_subwords]
                all_ngrams.update(ngrams)
                url_in_ngrams.append(ngrams)
                all_words.add("<UNKNOWN>")
                url_in_words.append("<UNKNOWN>")
            else:
                all_ngrams.update(ngrams)
                url_in_ngrams.append(ngrams)
                all_words.add(word)
                url_in_words.append(word)
        ngramed_x.append(url_in_ngrams)
        worded_x.append(url_in_words)

    ngrams_dict = {ngram: idx+1 for idx, ngram in enumerate(sorted(all_ngrams))}
    print("Size of ngram vocabulary: {}".format(len(ngrams_dict)))
    words_dict = {word: idx+1 for idx, word in enumerate(sorted(all_words))}
    print("Size of word vocabulary: {}".format(len(words_dict)))
    print("Index of <UNKNOWN> word: {}".format(words_dict.get("<UNKNOWN>", "Not found")))

    ngramed_id_x = []
    for ngramed_url in ngramed_x:
        url_in_ngrams = []
        for ngramed_word in ngramed_url:
            ngram_ids = [ngrams_dict.get(ngram, 0) for ngram in ngramed_word]
            url_in_ngrams.append(ngram_ids)
        ngramed_id_x.append(url_in_ngrams)

    worded_id_x = []
    for worded_url in worded_x:
        word_ids = [words_dict.get(word, 0) for word in worded_url]
        worded_id_x.append(word_ids)

    return ngramed_id_x, ngrams_dict, worded_id_x, words_dict

def ngram_id_x_from_dict(word_x, max_len_subwords, ngram_dict, word_dict=None):
    char_ngram_len = 1
    ngramed_id_x = []
    worded_id_x = []
    word_vocab = set(word_dict.keys()) if word_dict else None

    for url in word_x:
        url_in_ngrams = []
        url_in_words = []
        for word in url:
            ngrams = get_char_ngrams(char_ngram_len, word)
            if len(ngrams) > max_len_subwords:
                ngrams = ngrams[:max_len_subwords]
                word = "<UNKNOWN>"

            ngrams_id = [ngram_dict.get(ngram, 0) for ngram in ngrams]
            url_in_ngrams.append(ngrams_id)

            if word_dict:
                word_id = word_dict.get(word, word_dict.get("<UNKNOWN>", 0))
            else:
                word_id = 0
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

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def save_test_result(labels, all_predictions, all_scores, output_dir):
    output_labels = [1 if i == 1 else -1 for i in labels]
    output_preds = [1 if i == 1 else -1 for i in all_predictions]
    softmax_scores = [softmax(score) for score in all_scores]
    with open(output_dir, "w") as file:
        file.write("label\tpredict\tscore\n")
        for label, pred, score in zip(output_labels, output_preds, softmax_scores):
            file.write(f"{label}\t{pred}\t{score[1]}\n")

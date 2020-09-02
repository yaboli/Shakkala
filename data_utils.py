import os
import pickle
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
import numpy as np

VERSION_MAX_LENGTH_MAP = {
    '1': 495,
    '2': 315,
    '3': 315
}


def load_binary(file, folder):
    location = os.path.join(folder, (file + '.pickle'))
    with open(location, 'rb') as ff:
        data = pickle.load(ff)
    return data


def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = [line.strip() for line in f.readlines() if line.strip()]
    return content


def invert_map(original_map):
    return {v: k for k, v in original_map.items()}


class Dataset(object):
    def __init__(self, root='./', version=3):
        self.max_length = VERSION_MAX_LENGTH_MAP[str(version)]
        self.root = root
        dictionary_folder = os.path.join(self.root, 'dictionary')
        self.token2idx = load_binary('input_vocab_to_int', dictionary_folder)
        self.idx2tag = load_binary('output_int_to_vocab', dictionary_folder)
        self.tag2idx = invert_map(self.idx2tag)

    def build_dataset(self, filename, test_size=0.1, shuffle=True):
        data_folder = os.path.join(self.root, 'data')
        file_path = os.path.join(data_folder, filename)
        raw = read_file(file_path)
        tokens_padded, tags_padded = self.process_raw(raw)
        train_size = 1.0 - test_size
        train_tokens, test_tokens, train_tags, test_tags = train_test_split(tokens_padded, tags_padded,
                                                                            test_size=test_size, train_size=train_size,
                                                                            shuffle=shuffle)
        return train_tokens, test_tokens, train_tags, test_tags

    def process_raw(self, raw):
        tokens = list()
        tags = list()
        for line in raw:
            # print(line)
            length = len(line)
            if length >= self.max_length:
                continue
            token_idx = list()
            tag_idx = list()
            for i in range(length):
                char = line[i]
                if char in self.token2idx:  # current char is a valid token
                    token_idx.append(self.token2idx[char])
                    if i < length - 1:  # current char is in the middle
                        next_char = line[i + 1]
                        if next_char in self.tag2idx:  # next char is a valid tag
                            tag_idx.append(self.tag2idx[next_char])
                            i += 1  # skip next char
                        else:
                            tag_idx.append(self.tag2idx['ـ'])
                    else:  # current char is at the end, meaning no diacritic mark follows
                        tag_idx.append(self.tag2idx['ـ'])

            assert len(token_idx) == len(tag_idx), 'Number of tokens must equal number of tags!'

            # print(token_idx)
            # print(tag_idx)

            tokens.append(token_idx)
            tags.append(tag_idx)

        tokens_padded = pad_sequences(tokens, maxlen=self.max_length, dtype='int32', padding='post',
                                      value=self.token2idx['<PAD>'])
        tags_padded = pad_sequences(tags, maxlen=self.max_length, dtype='int32', padding='post',
                                    value=self.tag2idx['<PAD>'])
        tags_padded = np.expand_dims(tags_padded, axis=-1)

        return tokens_padded, tags_padded


if __name__ == '__main__':
    ds = Dataset()
    x, _, y, _ = ds.build_dataset('test_small.txt')
    x = x[:5]
    y = y[:5]
    for tokens, tags in zip(x, y):
        print(tokens)
        print(tags)

from collections import Counter
from itertools import chain
import torch
from typing import List
from utils import read_corpus, pad_sents, read_corpus_char, read_list_char

class Vocab(object):
    """ Vocabulary Entry, i.e. structure containing either
    src or tgt language terms.
    """
    def __init__(self, word2id=None):
        """ Init Vocab Instance.
        @param word2id (dict): dictionary mapping words 2 indices
        """
        if word2id:
            self.word2id = word2id
        else:
            self.word2id = dict()
            self.word2id['<pad>'] = 0   # Pad Token
            self.word2id['<unk>'] = 1   # Unknown Token
            self.word2id['<uppercase>'] = 2
        self.unk_id = self.word2id['<unk>']
        self.id2word = {v: k for k, v in self.word2id.items()}

    def __getitem__(self, word):
        """ Retrieve word's index. Return the index for the unk
        token if the word is out of vocabulary.
        @param word (str): word to look up.
        @returns index (int): index of word
        """
        try:
            idx = self.word2id[word]
            return idx
        except:
            if word.isupper():
                return self.word2id['<uppercase>']
            else:
                return self.unk_id

    def __contains__(self, word):
        """ Check if word is captured by Vocab.
        @param word (str): word to look up
        @returns contains (bool): whether word is contained
        """
        return word in self.word2id

    def __setitem__(self, key, value):
        """ Raise error, if one tries to edit the Vocab.
        """
        raise ValueError('vocabulary is readonly')

    def __len__(self):
        """ Compute number of words in Vocab.
        @returns len (int): number of words in Vocab
        """
        return len(self.word2id)

    def __repr__(self):
        """ Representation of Vocab to be used
        when printing the object.
        """
        return 'Vocabulary[size=%d]' % len(self)

    def id2word(self, wid):
        """ Return mapping of index to word.
        @param wid (int): word index
        @returns word (str): word corresponding to index
        """
        return self.id2word[wid]

    def add(self, word):
        """ Add word to Vocab, if it is previously unseen.
        @param word (str): word to add to Vocab
        @return index (int): index that the word has been assigned
        """
        if word not in self:
            wid = self.word2id[word] = len(self)
            self.id2word[wid] = word
            return wid
        else:
            return self[word]

    def words2indices(self, sents):
        """ Convert list of words or list of sentences of words
        into list or list of list of indices.
        @param sents (list[str] or list[list[str]]): sentence(s) in words
        @return word_ids (list[int] or list[list[int]]): sentence(s) in indices
        """
        if type(sents[0]) == list:
            return [[self[w] for w in s] for s in sents]
        else:
            return [self[w] for w in sents]

    def indices2words(self, word_ids):
        """ Convert list of indices into words.
        @param word_ids (list[int]): list of word ids
        @return sents (list[str]): list of words
        """
        return [self.id2word[w_id] for w_id in word_ids]

    def to_input_tensor(self, sents: List[List[str]], device: torch.device) -> torch.Tensor:
        """ Convert list of sentences (words) into tensor with necessary padding for
        shorter sentences.
        @param sents (List[List[str]]): list of sentences (words)
        @param device: device on which to load the tesnor, i.e. CPU or GPU
        @returns sents_var: tensor of (max_sentence_length, batch_size)
        """
        word_ids = self.words2indices(sents)
        sents_t = pad_sents(word_ids, self['<pad>'])
        sents_var = torch.tensor(sents_t, dtype=torch.long, device=device)
        return torch.t(sents_var)

    @staticmethod
    def from_corpus(path, size, freq_cutoff=5):
        """ Given a corpus construct a Vocab Entry.
        @param file_path (str): path to file containing corpus
        @param size (int): # of words in vocabulary
        @param freq_cutoff (int): if word occurs n < freq_cutoff times, drop the word
        @returns vocab_entry (Vocab): Vocab instance produced from provided corpus
        """
        corpus = read_corpus(path)
        vocab_entry = Vocab()
        word_freq = Counter(chain(*corpus))
        valid_words = [w for w, v in word_freq.items() if v >= freq_cutoff]
        #print('number of word types: {}, number of word types w/ frequency >= {}: {}'
         #     .format(len(word_freq), freq_cutoff, len(valid_words)))
        top_k_words = sorted(valid_words, key=lambda w: word_freq[w], reverse=True)[:size]
        for word in top_k_words:
            vocab_entry.add(word)
        print(len(vocab_entry))
        return vocab_entry

    @staticmethod
    def from_list(path, size, freq_cutoff=5):
        """ Given a corpus construct a Vocab Entry.
        @param file_path (str): variable to file containing list
        @param size (int): # of words in vocabulary
        @param freq_cutoff (int): if word occurs n < freq_cutoff times, drop the word
        @returns vocab_entry (Vocab): Vocab instance produced from provided corpus
        """
        corpus = []
        for line in path:
            sent = line.strip().split()
            corpus.append(sent)

        vocab_entry = Vocab()
        word_freq = Counter(chain(*corpus))
        valid_words = [w for w, v in word_freq.items() if v >= freq_cutoff]
        #print('number of word types: {}, number of word types w/ frequency >= {}: {}'
        #      .format(len(word_freq), freq_cutoff, len(valid_words)))
        top_k_words = sorted(valid_words, key=lambda w: word_freq[w], reverse=True)[:size]
        for word in top_k_words:
            vocab_entry.add(word)
        return vocab_entry


class Subword_vocab(Vocab):
    @staticmethod
    def from_corpus(path):
        corpus = read_corpus_char(path)
        vocab_entry = Subword_vocab()
        word_freq = Counter(chain(*corpus))
        valid_words = [w for w, v in word_freq.items()]
        for word in valid_words:
            vocab_entry.add(word)
        print(len(vocab_entry))
        return vocab_entry
    @staticmethod
    def from_list(path):
        corpus = read_list_char(path)
        vocab_entry = Subword_vocab()
        word_freq = Counter(chain(*corpus))
        valid_words = [w for w, v in word_freq.items()]
        for word in valid_words:
            vocab_entry.add(word)
        return vocab_entry

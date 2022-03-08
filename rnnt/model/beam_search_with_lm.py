import os


class BaseLM:
    def __init__(self):
        print("Overiding BaseLM")

    def get_first_state(self):
        raise NotImplementedError

    def update(self, state, word):
        raise NotImplementedError

    def get_word_score(self, state, word):
        raise NotImplementedError

    def get_sentence_score(self, sentence):
        raise NotImplementedError


class PinyinDict:
    def __init__(self):
        pass

    def pinyin2words(self, pinyin):
        return []


class NgramLM(BaseLM):
    def __init__(self):
        pass

    def get_first_state(self):
        return 0

    def update(self, state, word):
        new_state = state + 1
        return new_state

    def get_word_score(self, state, word):
        return 1.0

    def get_sentence_score(self, sentence):
        return 1.0
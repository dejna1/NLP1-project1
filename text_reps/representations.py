import numpy as np
from tokenizers import Tokenizer


class Encoder:
    """
    Base class, same interface as the tutorial's Encoder,
    so this file works on its own and the encoders still fit the tutorial's code.
    """

    def __len__(self) -> int:
        """Dimensionality of the representation space"""
        raise NotImplementedError

    def __call__(self, x: str) -> np.ndarray:
        """Represent x as a D-dimensional vector of real numbers"""
        raise NotImplementedError


class TokenIndicator(Encoder):
    """
    Same as the tutorial's TokenIndicator, with one fix:
    the tutorial's __len__ uses a global variable `tokenizer`
    instead of self.tokenizer, so it only works if a variable with
    that exact name happens to exist in the notebook.
    """

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return self.tokenizer.get_vocab_size()

    def __call__(self, x: str) -> np.ndarray:
        token_ids = self.tokenizer.encode(x).ids
        vector = np.zeros(len(self))
        vector[list(set(token_ids))] = 1      
        return vector


class BagOfWords(Encoder):
    """
    Like TokenIndicator, but each dimension counts how often its token occurs.
    'the cat saw the dog' -> the dimension for 'the' gets 2 instead of 1.
    """

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return self.tokenizer.get_vocab_size()

    def __call__(self, x: str) -> np.ndarray:
        token_ids = self.tokenizer.encode(x).ids
        vector = np.zeros(len(self))
        np.add.at(vector, token_ids, 1)
        return vector


class TfIdf(BagOfWords):
    """
    TF-IDF = term frequency x inverse document frequency.

    tf(token, text) = how often the token occurs in this text (the bag-of-words count)
    idf(token)      = log((1 + N) / (1 + df(token))) + 1
                      N  = number of texts used to fit the encoder
                      df = number of those texts that contain the token
                      (the +1s avoid dividing by zero; same smoothing as scikit-learn)
    """

    def __init__(self, tokenizer: Tokenizer, texts: list[str], normalise: bool = True):
        super().__init__(tokenizer)
        self.normalise = normalise

        document_frequency = np.zeros(len(self))
        for text in texts:
            unique_ids = list(set(self.tokenizer.encode(text).ids))
            document_frequency[unique_ids] += 1

        num_texts = len(texts)
        self.idf = np.log((1 + num_texts) / (1 + document_frequency)) + 1

    def __call__(self, x: str) -> np.ndarray:
        counts = super().__call__(x)          # the bag-of-words vector (tf)
        vector = counts * self.idf
        if self.normalise:
            # scale to length 1, so long and short tweets end up on the same scale
            length = np.linalg.norm(vector)
            if length > 0:
                vector = vector / length
        return vector
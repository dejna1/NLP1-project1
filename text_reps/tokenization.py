from pathlib import Path
from tokenizers import Tokenizer, models, pre_tokenizers, trainers


def train_tokenizer(texts: list[str], vocab_size: int = 1000,
                    min_frequency: int = 2):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
    )
    tokenizer.train_from_iterator(texts, trainer)
    return tokenizer


def build_tokenizer(X_train, vocab_size, min_frequency):
    tokenizer = train_tokenizer(X_train, vocab_size=vocab_size, min_frequency=min_frequency)
    return tokenizer


def save_tokenizer(tokenizer, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(path))


def load_tokenizer(path):
    return Tokenizer.from_file(str(path))
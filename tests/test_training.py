from argubot.generate.training import IGNORE_INDEX, encode_example, pad_batch


class FakeTokenizer:
    """Whitespace tokenizer. EOS and pad share id 0, like the GPT-2 habit in the base notebook."""

    eos_token_id = 0

    def __init__(self):
        self.vocab = {}

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [self.vocab.setdefault(w, len(self.vocab) + 1) for w in text.split()]}


def test_prompt_tokens_are_masked_and_reply_tokens_are_kept():
    tok = FakeTokenizer()
    enc = encode_example(tok, "a b c\n", "x y", max_len=20)
    assert enc["labels"][:3] == [IGNORE_INDEX] * 3
    assert enc["labels"][3:] == enc["input_ids"][3:]
    assert len(enc["attention_mask"]) == len(enc["input_ids"])


def test_eos_stays_in_the_labels_even_when_it_equals_the_pad_id():
    enc = encode_example(FakeTokenizer(), "a b\n", "x", max_len=20)
    assert enc["labels"][-1] == 0


def test_long_examples_drop_the_oldest_prompt_tokens_first():
    enc = encode_example(FakeTokenizer(), "a b c d e\n", "x y", max_len=4)
    assert len(enc["input_ids"]) == 4
    assert enc["labels"][-3:] == enc["input_ids"][-3:]


def test_a_reply_longer_than_the_limit_is_cut_not_the_other_way_round():
    enc = encode_example(FakeTokenizer(), "a b\n", "x y z w", max_len=3)
    assert len(enc["input_ids"]) == 3
    assert all(label != IGNORE_INDEX for label in enc["labels"])


def test_padding_masks_labels_and_attention():
    f1 = {"input_ids": [1, 2, 3], "labels": [-100, 2, 3], "attention_mask": [1, 1, 1]}
    f2 = {"input_ids": [4], "labels": [4], "attention_mask": [1]}
    batch = pad_batch([f1, f2], pad_id=0)
    assert batch["input_ids"][1] == [4, 0, 0]
    assert batch["labels"][1] == [4, IGNORE_INDEX, IGNORE_INDEX]
    assert batch["attention_mask"][1] == [1, 0, 0]

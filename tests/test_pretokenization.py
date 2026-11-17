
import pytest

from cs336_basics.bpe.pretokenization import split_special_tokens


@pytest.mark.parametrize(
    "text, special_tokens, expected",
    [
        (
            "Hello World",
            ["<|endoftext|>"],
            [("Hello World", False)],
        ),
        (
            "<|endoftext|>Hello",
            ["<|endoftext|>"],
            [("<|endoftext|>", True), ("Hello", False)],
        ),
        (
            "Hello<|endoftext|>",
            ["<|endoftext|>"],
            [("Hello", False), ("<|endoftext|>", True)],
        ),
        (
            "A<|endoftext|><|endoftext|>B",
            ["<|endoftext|>"],
            [
                ("A", False),
                ("<|endoftext|>", True),
                ("<|endoftext|>", True),
                ("B", False),
            ],
        ),
        (
            "A<|endoftext|>B<|pad|>C",
            ["<|endoftext|>", "<|pad|>"],
            [
                ("A", False),
                ("<|endoftext|>", True),
                ("B", False),
                ("<|pad|>", True),
                ("C", False),
            ],
        ),
        (
            "Hello World",
            [],
            [("Hello World", False)],
        ),
    ],
)
def test_split_special_tokens(text, special_tokens, expected):
    result = list(split_special_tokens(text, special_tokens))
    assert result == expected

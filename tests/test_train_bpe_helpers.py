
from cs336_basics.bpe.train_bpe import merge_pair


def test_merge_repeated_pairs():
    pretoken_freq = {
        (b"a", b"b", b"a", b"b", b"c"): 2,
    }

    result = merge_pair(pretoken_freq, (b"a", b"b"))

    assert result == {
        (b"ab", b"ab", b"c"): 2,
    }


def test_merge_overlapping_pairs():
    pretoken_freq = {
        (b"a", b"a", b"a"): 2,
    }

    result = merge_pair(pretoken_freq, (b"a", b"a"))

    assert result == {
        (b"aa", b"a"): 2,
    }


def test_merge_key_collision():
    pretoken_freq = {
        (b"a", b"b"): 3,
        (b"ab",): 2,
    }

    result = merge_pair(pretoken_freq, (b"a", b"b"))

    assert result == {
        (b"ab",): 5,
    }

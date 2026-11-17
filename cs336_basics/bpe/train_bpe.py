from collections import defaultdict
from pathlib import Path

from cs336_basics.bpe.pretokenization import iter_pretokens,split_special_tokens


# ============================================================
# 1. Vocabulary Initialization
# ============================================================

def initialize_vocab(
    special_tokens: list[str],
) -> dict[int, bytes]:
    """
    Initialize the BPE vocabulary.

    Initial vocabulary contains:
        - 256 possible byte values
        - All special tokens

    Returns:
        dict[int, bytes]: token ID -> token bytes
    """
    vocab = {}
    for i in range(256):
        vocab[i] = bytes([i])

    next_i = 256
    for token in special_tokens:
        vocab[next_i] = token.encode()
        next_i += 1

    return vocab


# ============================================================
# 2. Pre-tokenization and Frequency Counting
# ============================================================

def get_pretoken_freq(
    input_path: str | Path,
    special_tokens: list[str] | None,
) -> dict[tuple[bytes, ...], int]:
    """
    Read the training corpus, pre-tokenize it,
    convert pre-tokens into UTF-8 byte sequences,
    and count their frequencies.

    Returns:
        dict[tuple[bytes, ...], int]:
            Byte-token sequence -> frequency
    """

    # 1. Locate the project root
    root_dir = Path(__file__).resolve().parent.parent.parent

    # 2. Resolve the input path
    input_path = Path(input_path)
    data_path = (
        input_path if input_path.is_absolute()
        else root_dir / input_path
    )

    # 3. Read the training corpus
    text = data_path.read_text(encoding="utf-8")

    # 4. Initialize frequency dictionary
    pretoken_freq = defaultdict(int)

    # 5. Pre-tokenize and count frequencies
    for segment, is_special in split_special_tokens(
            text,
            special_tokens,
        ):

        if is_special: 
            continue

        for pretoken in iter_pretokens(segment):

            # Convert string to UTF-8 bytes
            pretoken_bytes = pretoken.encode("utf-8")

            # Split into individual byte tokens
            byte_tokens = tuple(
                bytes([b]) for b in pretoken_bytes
            )

            # Update frequency
            pretoken_freq[byte_tokens] += 1

    return dict(pretoken_freq)


# ============================================================
# 3. Pair Frequency Counting
# ============================================================

def get_pair_counts(
    pretoken_freq: dict[tuple[bytes, ...], int],
) -> dict[tuple[bytes, bytes], int]:
    """
    Count the frequency of every adjacent token pair.

    Returns:
        dict[tuple[bytes, bytes], int]:
            Token pair -> frequency
    """
    pair_counts = defaultdict(int)

    for pretoken, num in pretoken_freq.items():

        for pair_token in tuple(zip(pretoken[:], pretoken[1:])):
            pair_counts[pair_token] += num

    return dict(pair_counts)


# ============================================================
# 4. Select the Most Frequent Pair
# ============================================================

def get_most_frequent_pair(
    pair_counts: dict[tuple[bytes, bytes], int],
) -> tuple[bytes, bytes] | None:
    """
    Select the most frequent pair.

    Tie-breaking:
        Prefer the lexicographically greater pair.

    Returns:
        The selected pair, or None if no pair exists.
    

    max(iterable, key=func)
    example:
    return max(
        pair_counts,
        key=lambda pair: (pair_counts[pair], pair)
    )

    """

    if not pair_counts:
        return None
    
    most_frequent_pair = None
    max_count = -1

    for pair, num in pair_counts.items():

        if num > max_count:
            most_frequent_pair = pair
            max_count = num

        elif num == max_count and pair > most_frequent_pair:
                most_frequent_pair = pair
            
    return most_frequent_pair


# ============================================================
# 5. Merge the Selected Pair
# ============================================================

def merge_pair(
    pretoken_freq: dict[tuple[bytes, ...], int],
    pair_to_merge: tuple[bytes, bytes],
) -> dict[tuple[bytes, ...], int]:
    """
    Merge every non-overlapping occurrence of the target pair
    in all pre-tokens.

    Returns:
        Updated pre-token frequency dictionary.
    """
    merge_freq = defaultdict(int)

    for pretoken, num in pretoken_freq.items():
        list_pretoken = []
        i = 0
        while(i < len(pretoken)):
            if (i + 1 < len(pretoken) and (pretoken[i], pretoken[i + 1]) == pair_to_merge):
                list_pretoken.append(pretoken[i] + pretoken[i + 1])
                i += 2
            else: 
                list_pretoken.append(pretoken[i])
                i += 1

        new_pretoken = tuple(list_pretoken)
        merge_freq[new_pretoken] += num

    return dict(merge_freq)


# ============================================================
# 6. Main BPE Training Function
# ============================================================

def train_bpe(
    input_path: str | Path,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a byte-level BPE tokenizer.

    Args:
        input_path: Training corpus path.
        vocab_size: Target vocabulary size.
        special_tokens: Special tokens to include in vocabulary.
        **kwargs: Optional additional arguments.

    Returns:
        vocab:
            Token ID -> token bytes

        merges:
            Ordered list of merged token pairs.
    """
    merged_token_pairs = []
    vocab = initialize_vocab(special_tokens=special_tokens)
    pretoken_freq = get_pretoken_freq(input_path=input_path, special_tokens=special_tokens)

    while len(vocab) < vocab_size:
        pair_counts = get_pair_counts(pretoken_freq=pretoken_freq)
        most_frequent_pair = get_most_frequent_pair(pair_counts=pair_counts)
        if most_frequent_pair is None:
            break

        vocab[len(vocab)] = most_frequent_pair[0] + most_frequent_pair[1]
        merged_token_pairs.append(most_frequent_pair)
        pretoken_freq = merge_pair(pretoken_freq=pretoken_freq, pair_to_merge=most_frequent_pair)

    return vocab, merged_token_pairs

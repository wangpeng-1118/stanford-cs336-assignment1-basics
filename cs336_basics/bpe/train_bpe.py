from collections import defaultdict
from pathlib import Path
from typing import Counter

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
    pretoken_index: dict[int, tuple[bytes, ...]],
    freq_index: dict[int, int],
    pair_to_merge: tuple[bytes, bytes],
    pair_to_token_index: dict[tuple[bytes, bytes], set[int]],
    pair_counts: dict[tuple[bytes, bytes], int]
):
    """
    Merge every non-overlapping occurrence of the target pair
    in all pre-tokens.

    Returns:
        Updated pre-token frequency dictionary.
    """
    if pair_to_merge not in pair_to_token_index:
        return

    # new_pair_counts = defaultdict(int, pair_counts)
    # new_pair_to_token_index = defaultdict(set, pair_to_token_index)
    token_indices = pair_to_token_index[pair_to_merge].copy()

    for idx in token_indices:

        pretoken = pretoken_index[idx]
        freq = freq_index[idx]

        old_pairs = Counter(
            zip(pretoken, pretoken[1:])
        )

        for current_pair, count in old_pairs.items():
            pair_to_token_index[current_pair].discard(idx)

            if not pair_to_token_index[current_pair]:
                del pair_to_token_index[current_pair]

            pair_counts[current_pair] -= count * freq
            assert pair_counts[current_pair] >= 0
            if pair_counts[current_pair] == 0:
                del pair_counts[current_pair]

        # Merge the target pair in the pre-token
        merged_pretoken = []
        i = 0
        while i < len(pretoken):
            if (i < len(pretoken) - 1 and
                    pretoken[i] == pair_to_merge[0] and
                    pretoken[i + 1] == pair_to_merge[1]):
                merged_pretoken.append(pair_to_merge[0] + pair_to_merge[1])
                i += 2
            else:
                merged_pretoken.append(pretoken[i])
                i += 1

        # Update the pre-token index and frequency index
        pretoken_index[idx] = tuple(merged_pretoken)
        freq_index[idx] = freq

        for i in range(len(merged_pretoken) - 1):
            new_pair = (merged_pretoken[i], merged_pretoken[i + 1])
            # pair_to_token_index[new_pair] = pair_to_token_index.get(new_pair, set()).union({idx})
            pair_to_token_index.setdefault(new_pair, set()).add(idx)
            pair_counts[new_pair] = pair_counts.get(new_pair, 0) + freq

    return


def get_pair_to_token_index(
    pretoken_index: dict[int, tuple[bytes, ...]],
) -> dict[tuple[bytes, bytes], set[int]]:
    """
    Create a mapping from each token pair to the indices of pre-tokens
    that contain that pair.

    Returns:
        dict[tuple[bytes, bytes], set[int]]:
            Token pair -> set of pre-token indices
    """
    pair_to_token_index = defaultdict(set)

    for idx, pretoken in pretoken_index.items():

        for pair in zip(pretoken[:], pretoken[1:]):
            pair_to_token_index[pair].add(idx)

    return dict(pair_to_token_index)


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
    pretoken_index = {idx : pretoken for idx, pretoken in enumerate(pretoken_freq.keys())}
    freq_index = {idx : freq for idx, freq in enumerate(pretoken_freq.values())}

    pair_counts = get_pair_counts(pretoken_freq=pretoken_freq)
    pair_to_token_index = get_pair_to_token_index(pretoken_index=pretoken_index)
    while len(vocab) < vocab_size:
        
        most_frequent_pair = get_most_frequent_pair(pair_counts=pair_counts)
        if most_frequent_pair is None:
            break

        merged_token_pairs.append(most_frequent_pair)
        merge_pair(
            pretoken_index=pretoken_index,
            freq_index=freq_index,
            pair_to_token_index=pair_to_token_index,
            pair_to_merge=most_frequent_pair,
            pair_counts=pair_counts)

        vocab[len(vocab)] = most_frequent_pair[0] + most_frequent_pair[1]

    return vocab, merged_token_pairs

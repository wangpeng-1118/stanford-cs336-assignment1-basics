import regex

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def iter_pretokens(text: str):
    """
    将普通文本切分成 Pre-tokens。

    Args:
        text: 输入的普通文本。

    Yields:
        一个个 Pre-token 字符串。
    """

    for match in regex.finditer(PAT, text):
        yield match.group()


def split_special_tokens(
    text: str,
    special_tokens: list[str],
):
    """
    按 Special Tokens 划分文本。

    Yields:
        tuple[str, bool]:
            str: 文本片段
            bool: True 表示 Special Token，
                  False 表示普通文本

    # example
    for segment, is_special in split_special_tokens(
        text,
        special_tokens,
    ):
    """
    if not special_tokens:
        if text:
            yield text, False
        return

    # 优先匹配长的special tokens，避免短特殊字符是长特殊字符的前缀
    special_tokens = sorted(
        set(special_tokens),
        key=len,
        reverse=True,
    )

    if any(token == "" for token in special_tokens):
        raise ValueError("Special tokens cannot be empty")

    # 2. Escape special tokens and construct a regex pattern
    # join() ：string function
    pattern = "|".join(
        regex.escape(token)
        for token in special_tokens
    )

    # 3. Find all special tokens
    last_end = 0

    for match in regex.finditer(pattern, text):

        # 4. Yield ordinary text before the special token
        if last_end < match.start():
            yield text[last_end:match.start()], False

        # 5. Yield the special token itself
        yield match.group(), True

        last_end = match.end()

    # 6. Yield remaining ordinary text
    if last_end < len(text):
        yield text[last_end:], False



    

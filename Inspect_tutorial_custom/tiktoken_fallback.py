"""
tiktoken_fallback.py
--------------------
Monkey-patches inspect_ai's token counter to use a character-based
approximation when the tiktoken vocab file cannot be downloaded
(e.g. restricted network environments, CI, air-gapped machines).

Import this BEFORE running any inspect_ai code:

    import tiktoken_fallback  # noqa: F401
    from inspect_ai import task, eval, Task
    ...

Or prepend it to a script:

    python3 -c "import tiktoken_fallback" && inspect eval my_task.py --model mockllm/model

You do NOT need this on a normal internet-connected machine — tiktoken
will download and cache the vocab file automatically on first use.
"""


def _patched_count_text_tokens(text: str) -> int:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("o200k_base")
        token_count = len(enc.encode(text, disallowed_special=()))
        return max(1, int(token_count * 1.1))
    except Exception:
        # Fallback: ~4 chars per token with 10% buffer
        return max(1, int(len(text) / 4 * 1.1))


import inspect_ai.model._tokens as _tokens  # noqa: E402

_tokens.count_text_tokens = _patched_count_text_tokens

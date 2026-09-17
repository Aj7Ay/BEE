from __future__ import annotations


def sanitize_for_terminal(value: str) -> str:
    """Escape C0 control characters and DEL so a filesystem-controlled
    string (a path, in practice) can never inject terminal control
    sequences into text-mode output -- an attacker who controls a
    filename in a scanned directory could otherwise recolor text, move
    the cursor, or overwrite a real finding line with a fabricated
    "clean" one on the operator's terminal. JSON output is untouched:
    it isn't interpreted by a terminal, and the raw bytes may matter to
    a downstream consumer.
    """
    return "".join(
        f"\\x{ord(ch):02x}" if ord(ch) < 0x20 or ord(ch) == 0x7F else ch
        for ch in value
    )

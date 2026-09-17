from bee.cli.sanitize import sanitize_for_terminal


def test_escapes_ansi_escape_byte():
    result = sanitize_for_terminal("esc\x1b[31mred")
    assert "\x1b" not in result
    assert "\\x1b" in result


def test_escapes_all_c0_control_and_del():
    raw = "".join(chr(c) for c in range(0x20)) + "\x7f"
    result = sanitize_for_terminal(raw)
    assert all(ord(ch) >= 0x20 and ord(ch) != 0x7f for ch in result)


def test_leaves_ordinary_text_unchanged():
    assert sanitize_for_terminal("models/weights.gguf") == "models/weights.gguf"


def test_leaves_rich_markup_brackets_as_literal_text():
    # sanitize_for_terminal only neutralizes raw control bytes -- Rich
    # markup injection ("[bold red]...") is handled separately by
    # rendering untrusted strings through rich.text.Text, which never
    # interprets its content as markup regardless of what characters it
    # contains.
    assert sanitize_for_terminal("[bold red]x[/bold red]") == "[bold red]x[/bold red]"

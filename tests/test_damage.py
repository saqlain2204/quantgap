from quantgap.score import damage as d


def test_t7_loop_detection():
    assert d.is_loop(list(range(8)) * 5)  # an 8-token run repeated 5 times
    assert d.is_loop([1, 2, 3] * 30)  # few distinct tokens in the last 64
    assert not d.is_loop(list(range(200)))


def test_t7_wrong_script():
    assert d.has_wrong_script("The answer is 42 \u4f60", "The answer is 42")
    assert not d.has_wrong_script("The answer is 42", "The answer is 42")
    # the reference is allowed to use the script itself
    assert not d.has_wrong_script("\u4f60\u597d", "\u4f60\u597d hello")
    assert d.has_wrong_script("\u043f\u0440\u0438\u0432\u0435\u0442", "hello")


def test_t7_json_valid():
    assert d.json_valid('Sure! {"a": 1, "b": {"c": [1, 2]}} done')
    assert not d.json_valid('{"a": 1, "b": ')
    assert not d.json_valid("no json here")
    assert d.json_valid('{"s": "brace } inside"}')


def test_clean_text_flags_nothing():
    text = "Photosynthesis turns light into sugar."
    ids = list(range(100))
    assert not d.is_loop(ids)
    assert not d.has_wrong_script(text, text)
    assert not d.json_valid(text)


def test_first_divergence():
    assert d.first_divergence([1, 2, 3], [1, 2, 9]) == 2
    assert d.first_divergence([1, 2], [1, 2, 3]) == 2
    assert d.first_divergence([], [1]) == 0

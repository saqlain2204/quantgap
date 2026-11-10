import pytest

from quantgap.models import TokenizerInfo
from quantgap.tokcheck import TokenizerMismatch, TokFingerprint, check_tokenizer

PACK = TokenizerInfo(vocab_size=100, bos_id=1, eos_id=2, check_ids=[5, 6, 7, 8])


def test_ok():
    check_tokenizer(PACK, TokFingerprint(100, 1, 2, [5, 6, 7, 8]), pack_top_ids_max=99)


def test_t10_swapped_token_id_names_the_check_and_the_id():
    with pytest.raises(TokenizerMismatch, match=r"Check 2.*index 2.*id 7.*id 9"):
        check_tokenizer(PACK, TokFingerprint(100, 1, 2, [5, 6, 9, 8]), pack_top_ids_max=99)


def test_smaller_vocab_fails():
    with pytest.raises(TokenizerMismatch, match="Check 1"):
        check_tokenizer(PACK, TokFingerprint(90, 1, 2, [5, 6, 7, 8]), pack_top_ids_max=99)


def test_padding_ids_allowed():
    check_tokenizer(PACK, TokFingerprint(128, 1, 2, [5, 6, 7, 8]), pack_top_ids_max=99)


def test_bos_mismatch_fails():
    with pytest.raises(TokenizerMismatch, match="Check 3"):
        check_tokenizer(PACK, TokFingerprint(100, 3, 2, [5, 6, 7, 8]), pack_top_ids_max=99)


def test_server_unknown_vocab_skips_check_one():
    check_tokenizer(PACK, TokFingerprint(-1, None, None, [5, 6, 7, 8]), pack_top_ids_max=99)

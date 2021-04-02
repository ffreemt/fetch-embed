"""Test fetch_embed."""
from fetch_embed import fetch_embed


def test_fetch_embed():
    """Test fetch_embed."""
    texts = ["test 1", "测试1"]

    res = fetch_embed(texts, livepbar=False)

    assert res.shape == (2, 512)
    assert res.mean() > -0.1

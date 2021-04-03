"""Test fetch_embed."""
import numpy as np
from fetch_embed import fetch_embed


def test_fetch_embed():
    """Test fetch_embed."""
    texts = ["test 1", "测试1"]

    res = fetch_embed(texts, livepbar=False)

    assert np.array(res).shape == (2, 512)
    assert np.array(res).mean() > -0.1

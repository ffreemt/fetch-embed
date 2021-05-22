"""Test fetch_embed."""
import numpy as np
from fetch_embed import fetch_embed


def test_fetch_embed_la():
    """Test fetch_embed_la."""
    texts = ["test 1", "测试1"]

    ip_ = "127.0.0.1"
    endpoint = f"http://{ip_}/embed_la/"
    res = fetch_embed(texts, livepbar=False, endpoint=endpoint)

    # assert np.array(res).shape == (2, 512)
    assert np.array(res).shape == (2, 768)
    assert np.array(res).mean() > -0.1

"""Test embed_text."""
import shutil
import numpy as np
from fetch_embed import embed_text
shutil.rmtree("./joblibcache", ignore_errors=True)


def test_embed_text():
    """Test embed_text."""
    texts = ["test 1", "测试1"] * 17

    res = embed_text(texts)

    assert np.array(res).shape == (34, 512)
    assert np.array(res).mean() > -0.1

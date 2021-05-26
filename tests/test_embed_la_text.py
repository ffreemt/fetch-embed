"""Test embed_text."""
import shutil
import numpy as np
from fetch_embed import embed_text
shutil.rmtree("./joblibcache", ignore_errors=True)


def test_embed_la_text():
    """Test embed_text.

    ip_ = "127.0.0.1"
    endpoint = f"http://{ip_}:8000/embed_la/"
    ip_ = "embed.ttw.workder.dev"
    endpoint = f"https://{ip_}/embed_la/"

    la: language agnostic model (1.72G) dim-768
    """
    ip_ = "127.0.0.1"
    endpoint = f"http://{ip_}:8000/embed_la/"

    ip_ = "embed.ttw.workers.dev"
    endpoint = f"https://{ip_}/embed_la/"

    texts = ["test 1", "测试1"] * 18

    res = embed_text(texts, endpoint=endpoint)

    assert np.array(res).shape == (36, 768)
    assert np.array(res).mean() > -0.1

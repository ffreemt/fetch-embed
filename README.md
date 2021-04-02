# fetch-embed
<!--- fetch-embed  fetch_embed  fetch_embed fetch_embed --->
[![tests](https://github.com/ffreemt/fetch-embed/actions/workflows/routine-tests.yml/badge.svg)][![python](https://img.shields.io/static/v1?label=python+&message=3.7%2B&color=blue)](https://img.shields.io/static/v1?label=python+&message=3.7%2B&color=blue)[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)[![PyPI version](https://badge.fury.io/py/fetch_embed.svg)](https://badge.fury.io/py/fetch_embed)

fetch multilingual embed from embed.ttw.workers.de

## Install it
```bash
pip install -U fetch-embed
```

## Use it

```python
from fetch_embed import fetch_embed

res = fetch_embed("test me")
print(res.shape)
# (1, 512)

print(fetch_embed(["test me", "测试123"]).shape
# (2, 512)

# to turn off live progress bar
res = fetch_embed("test me", livepbar=False)

# brief docs
help(fetch_embed)
# fetch_embed(texts:Union[str, List[str]], endpoint:str='http://ttw.hopto.org/embed/', livepbar:bool=True) -> numpy.ndarray
    Fetch embed from endpoint.
```
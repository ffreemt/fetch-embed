"""Fetch embed from endpoint."""
from typing import List, Union

# from math import ceil
import httpx

from logzero import logger
from alive_progress import alive_bar

HOST1 = "ttw.hopto.org"
HOST2 = "embed.ttw.workers.dev"
EP1 = f"http://{HOST1}/embed/"
EP2 = f"http://{HOST2}/embed/"
CLIENT = httpx.Client()
try:
    httpx.get(EP1)
    EP_ = EP1
except Exception:
    EP_ = EP2


# fmt: off
def fetch_embed(
        texts: Union[str, List[str]],
        endpoint: str = EP_,
        livepbar: bool = True,  # need to turned off for pytest
        timeout: float = None,
        client=CLIENT,
) -> List[float]:
    """Fetch embed from endpoint."""
    if isinstance(texts, str):
        texts = [texts]
    data = {"text1": texts}

    resp = httpx.Response(200)

    def func_():
        nonlocal resp
        try:
            # resp = httpx.post(
            resp = client.post(
                endpoint,
                json=data,
                timeout=timeout)
            resp.raise_for_status()
        except Exception as exc:
            logger.error(exc)
            # msg = str(exc)
            raise

    if livepbar:
        with alive_bar(1, length=3) as pbar:
            func_()
            pbar()
    else:
        func_()

    try:
        jdata = resp.json()
    except Exception as exc:
        logger.error(exc)
        raise

    res = jdata.get("embed")
    if res is None:
        raise Exception("Cant get anything from jdata.get('embed'), probbaly wrong API...")

    # return np.array(res)
    return res

    # feed back error messages
    # return np.array([jdata.get("error")])

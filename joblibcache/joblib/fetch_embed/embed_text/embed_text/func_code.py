# first line: 16
@memory.cache
def embed_text(src_blocks: Union[str, List[str]], bsize:int = 32) -> np.ndarray:
  """Embed batch of text (bsize=32)."""
  if isinstance(src_blocks, str):
      src_blocks = [src_blocks]

  src_embed = []
  len_ = len(src_blocks)
  tot = len_ // bsize + bool(len_ % bsize)
  idx = 0
  pbar = tqdm(total=tot)
  for elm in mit.chunked(src_blocks, bsize):
      idx += 1
      logger.debug(" {}, {}", idx, idx / tot)
      try:
          _ = fetch_embed(elm, livepbar=False)
      except Exception as e:
          logger.debug(e)
          _ = [[str(e)] + [""] * 31]
      src_embed.extend(_)
      pbar.update()
  return np.array(src_embed)
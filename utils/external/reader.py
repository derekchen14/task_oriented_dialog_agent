"""
@authors: xiul, t-zalipt, antoinne
"""

def get_glove_name(opt, type_="tokens", key="pt"):
  emb_type = getattr(opt, key)
  if type_ == "tokens":
    return "data/{}/tokens.{}{}.vocab".format(emb_type, emb_type, opt.iSize)
  else:
    return "data/{}/entities.{}{}.vocab".format(emb_type, emb_type, opt.eSize)


def text_to_dict(path):
  """ Read in a text file as a dictionary
  where keys are text and values are indices (line numbers) """

  slot_set = {}
  with open(path, 'r') as f:
    index = 0
    for line in f.readlines():
      slot_set[line.strip('\n').strip('\r')] = index
      index += 1
  return slot_set
import re

splits = ["tst", "dev", "trn"]
total = []
# split = "trn"
for split in splits:
  path_name = "datasets/restaurants/dialog-babi-task6-dstc2-{}.txt".format(split)
  with open(path_name, "r") as f:
    for line in f:
      # tokens = line.split()
      turn = line.split('\t')
      try:
        response = turn[1]     # query = turn[0]
        # label = tokens[2]
        searchObj = re.search( r'no restaurant serving ([a-z]*) food', response)
        # if label == 'R_cuisine':
        if searchObj:
          # cuisine = tokens[3]
          cuisine = searchObj.group(1)
          total.append(cuisine)
      except(IndexError):
        continue
  # print set(total)
  total.sort()
  uniques = set(total)
  # print len(uniques)
  m = len(uniques)
  print "Finished processing set {0}, found {1} cuisines".format(split, m)


def identify_matches(sentence):
  matches = []
  for word in sentence:
  if index in match_phrases["cuisines"]:
    matches.append(0)
  elif index in match_phrases["locations"]:
    matches.append(1)
  elif index in match_phrases["prices"]:
    matches.append(2)
  elif index in match_phrases["ratings"]:
    matches.append(3)
  elif index in match_phrases["sizes"]:
    matches.append(4)
  elif index == PHONE_token:
    matches.append(5)
  elif index == ADDR_token:
    matches.append(6)
  else: # not a special match embedding
    matches.append(7)

  return matches
'''

def add_match_feature(sentence, use_cuda):
  matches = identify_matches(sentence)
  encoding_suffix = []
  for match in matches:
    thing = match_embeddings[match]
    encoding_suffix.extend(thing)
  # 1 = batch size / -1 = sentence length / 8 = size of match embedding
  reshaped = encoding_suffix.view(1,-1,8)
  # if use_cuda: reshaped.cuda()

  return sentence.extend(reshaped, axis=2)


'''
  # (10 choices, 91 in DTSC)
  cuisines = ["british", "cantonese", "french", "indian", "italian",
              "japanese", "korean", "spanish", "thai", "vietnamese"]
  # (10 choices, 5 in DTSC)
  locations = ["beijing", "seoul", "tokyo", "paris", "madrid",
              "hanoi", "bangkok", "rome", "london", "bombay"]
  prices = ["cheap", "moderate", "expensive"]
  ratings = [str(x) for x in range(1,9)]
  party_sizes = ["two", "four", "six", "eight"]

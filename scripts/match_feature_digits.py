import json
import torch
import utils.internal.vocabulary as vocab

# Run this file from the root folder, not from /scripts
vocab = json.load( open("datasets/res_vocab.json", "r") )

# short list include 10, full list has 24 cuisines, missing has 51 cuisines
# cuisines = ["british", "cantonese", "french", "indian", "italian",
#             "japanese", "korean", "spanish", "thai", "vietnamese"]
cuisines = ['portuguese', 'mexican', 'chinese', 'mediterranean', 'japanese',
            'spanish', 'vietnamese', 'gastropub', 'north_american', 'indian',
            'asian_oriental', 'international', 'korean', 'european', 'bistro',
            'french', 'fusion', 'thai', 'lebanese', 'seafood', 'turkish',
            'british', 'african', 'italian']
missing = ['irish', 'singaporean', 'bistro', 'german', 'cantonese',
            'australasian', 'kosher', 'moroccan', 'crossover', 'tuscan',
            'persian', 'halal', 'scottish', 'polish', 'corsica',
            'swedish', 'traditional', 'unusual', 'caribbean', 'romanian',
            'creative', 'brazilian', 'jamaican', 'australian', 'christmas',
            'belgian', 'danish', 'indonesian', 'austrian', 'hungarian',
            'welsh', 'panasian', 'catalan', 'fusion', 'polynesian', 'russian',
            'world', 'afghan', 'canapes', 'basque', 'cuban', 'vegetarian',
            'malaysian', 'scandinavian', 'venetian', 'greek', 'steakhouse',
            'english', 'swiss', 'barbeque']
locations = ["beijing", "seoul", "tokyo", "paris", "madrid",
            "hanoi", "bangkok", "rome", "london", "bombay"]
prices = ["cheap", "moderate", "expensive"]
sizes = ["two", "four", "six", "eight"]     # party size
PHONE_token = "<PHONE>"
ADDR_token = "<ADDR>"

vocab_size = len(vocab)
print("vocab_size: {}".format(vocab_size) )
match_features = torch.zeros((vocab_size, 8))

for idx, token in enumerate(vocab):
  if token in cuisines:
    match_features[idx, 0] = 1
  if token in missing:
    match_features[idx, 1] = 1
  if token in locations:
    match_features[idx, 2] = 1
  if token in prices:
    match_features[idx, 3] = 1
  if token in sizes:
    match_features[idx, 4] = 1
  if token == PHONE_token:
    match_features[idx, 5] = 1
  if token == ADDR_token:
    match_features[idx, 6] = 1
  else:
    match_features[idx, 7] = 1

torch.save(match_features, 'datasets/restaurants/match_features.pt')

'''
# ratings = [str(x) for x in range(1,9)]    # star rating

json.dump(phrases, open("phrases.json", "w"))
phrases = {"cuisines":[], "missing":[], "locations":[], "prices":[], "sizes":[]}
digit = res_vocab.index(token)
phrases["sizes"].append(digit)
print("{0}: {1}".format(token, digit) )

def add_match_feature(sentence, use_cuda):
  matches = identify_matches(sentence)
  encoding_suffix = []
  for match in matches:
    thing = match_embeddings[match]
    encoding_suffix.extend(thing)
  reshaped = encoding_suffix.view(1,-1,8)
  # if use_cuda: reshaped.cuda()

  return sentence.extend(reshaped, axis=2)
'''

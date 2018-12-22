import json
import pdb

split_types = ["test", "train"]
for split in split_types:
  data = json.load( open("clean_woz_{}_1.json".format(split), "rb"))
  cleaned = []
  counter = 0
  ex_id = 0

  for example in data:
    # cache user preference by selectively re-initializing for new examples
    if example['id'] != ex_id:
      frame = {}
      frame["food"] = "<NONE>"
      frame["price"] = "<NONE>" # cheap, moderate, expensive, dont_care
      frame["area"] = "<NONE>"  # north, south, east, west, centre, dont_care
      ex_id = example['id']
    # initialize the frame / clear out the previous frame
    frame["open"] = False
    frame["request"] = []     # postcode, address, price, area, food, hours, phone
    frame["clarify"] = False
    frame["accept"] = "<NONE>" # restaurant, response
    frame["reject"] = "<NONE>" # continue, end
    frame["close"] = False
    # could add turn number, NER, did we greet first, user sentiment, etc.

    for isv in example['intents']:
      frame["open"] = True if (isv[0] == "open" or frame["open"]) else False
      frame["close"] = True if (isv[0] == "close" or frame["close"]) else False
      frame["clarify"] = True if (isv[1] == "clarify" or frame["clarify"]) else False

      if (isv[0] == "inform" and isv[1] == "food"):
        frame["food"] = isv[2]
      if (isv[0] == "inform" and isv[1] == "price"):
        frame["price"] = isv[2]
      if (isv[0] == "inform" and isv[1] == "area"):
        frame["area"] = isv[2]
      if (isv[0] == "accept" and frame["accept"] == "<NONE>"):
        frame["accept"] = isv[2]
      if (isv[0] == "reject" and frame["accept"] == "<NONE>"):
        frame["reject"] = isv[1]
      if (isv[0] == "request" and isv[1] == "question"):
        frame["request"].append(isv[2])

    example['frame'] = frame.copy()
    cleaned.append(example)

    # if ex_id == 669:
    #   print(cleaned[-3]['frame']['close'])
    #   print(cleaned[-2]['frame']['close'])
    #   print(cleaned[-1]['frame']['close'])

  json.dump(cleaned, open("clean_woz_{}_4.json".format(split), "w", encoding="utf8"))
  print("Done with {}, processed {} examples".format(split, len(cleaned)))


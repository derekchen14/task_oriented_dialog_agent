import pdb, sys
import json
from random import shuffle
from collections import defaultdict
from nltk import word_tokenize

domain = "restaurants"  # ["movies", "taxis", "restaurants"]
slotvals = defaultdict(set)
slot_mapper = {
  'moviename': 'movie name',
  'ticket': 'ticket',
  'theater': 'theater',
  'genre': 'genre',
  'date': 'date',
  'restaurantname': 'restaurant name',
  'reservation': 'reservation',
  'atmosphere': 'atmosphere',
  'rating': 'rating',
  'city': 'city',
  'state': 'state',
  'starttime': 'start time',
  'numberofpeople': 'number of people',
  'price': 'price',
  'food': 'food',
  'car_type': 'car type',
  'pickup_time': 'pickup time',
  'pickup_city': 'pickup city',
  'dropoff_city': 'dropoff city',
  'pickup_location': 'pickup location',
  'dropoff_location': 'dropoff location',
  'ride': 'ride',
  'task': 'task',
  'other': 'other',
  'unknown': 'unknown',
}
act_mapper = {
  'open': 'open = hi hello',
  'close': 'close = goodbye thanks',
  'accept': 'accept = yes ok',
  'reject': 'reject = no not',
  'question': 'question = how what',
  'answer': 'answer = sure can',
  'unknown': 'unknown = other ',
  'multiple': 'multiple = available or',
}


def vectorize(text, style='intent'):
  # res = [word2index[word] for x in intent for word in x.split()]
  embedding = []
  embedding.append(word2index['<sos>'])


  if text[0] == '<special2>':
    text = "<special2>"
  elif style == 'intent':
    x, y = text
    if x == "request":
      slot = slot_mapper[y]
      text = f'{x} = {slot} <eos>'
    elif x == "act":
      text = act_mapper[y] + ' <eos>'
    else:  # its some type of inform
      slot, value = slot_mapper[x], y
      text = f'{slot} = {value} <eos>'

  for token in text.split():
    word = token.rstrip(',').rstrip(':').rstrip('.')
    try:
      embedding.append(word2index[word])
    except(KeyError):
      embedding.append(word2index['<unk>'])

  embedding.append(word2index['<eos>'])
  return embedding

def parse_intent(intent):
  other_acts = ['open', 'close', 'accept', 'reject', 'question', 'answer', 'unknown', 'multiple']
  items = {'taxis': ['ride'], 'movies': ['ticket', 'movie', 'time', 'theater'],
    'restaurants': ['reservation', 'restaurant', 'time', 'date'] }
  allowed = items[domain] + ['other']

  mark = intent.index('(')
  act = intent[:mark]
  if act == "inform":
    slot, value = intent[mark+1:-1].split('=')
    return [slot, value]
  elif act == "request":
    equal = intent.index('=')
    slot = intent[equal+1:-1]
    slot = slot if slot in allowed else 'unknown'
    return [act, slot]
  else:
    if act not in other_acts:
      act = "multiple"
    return ["act", act]

def find_beliefs(globe, current):
  beliefs = []
  for intent in current:
    if intent[0] == "request":
      beliefs.append({"act": intent[0], "slots": [intent]})
    elif intent[0] == "act":
      beliefs.append({"act": intent[1], "slots": [intent]})
  for s,v in globe.items():
    beliefs.append({"act": "inform", "slots": [[s, v]]})

  return beliefs

def update_global(global_beliefs, current):
  for intent in current:
    if intent[0] not in ['request', 'act']:
      slot, value = intent
      global_beliefs[slot] = value
  return global_beliefs

def add_to_collection(speaker, intents):
  if speaker == "user":
    for intent in intents:
      mark = intent.index('(')
      act = intent[:mark]
      if act == "inform":
        slot, value = intent[mark+1:-1].split('=')
        slotvals[slot].add(value)

if __name__ == "__main__":
  original = json.load( open(f'{domain}/original.json', 'r') )
  print("we started out with {} chats in {}".format(len(original), domain))
  revised = []

  vocab = json.load(open(f'{domain}/vocab.json', 'r'))
  index2word = vocab['index2word']
  word2index = {word: idx for idx, word in enumerate(index2word)}

  for i, chat in enumerate(original):
    dialogue = {'dialogue_id': i+1, 'turns': []}

    global_beliefs = {}  # only contains constraints from inform
    agent_utterance = ""
    agent_actions = [['<special2>']]

    for jdx, turn in enumerate(chat):
      unique_intents = list(set(turn['intents']))

      if turn["speaker"] == "agent":
        agent_utterance = turn['utterance']
        if len(unique_intents) > 0:
          agent_actions = [parse_intent(x) for x in unique_intents]

      elif turn["speaker"] == "user":
        new_turn = {}
        new_turn['turn_id'] = jdx
        new_turn['utterance'] = turn['utterance']
        current = [parse_intent(x) for x in unique_intents]
        if ['task', 'complete'] in current:
          current.remove(['task', 'complete'])
        new_turn['user_intent'] = current.copy()

        global_beliefs = update_global(global_beliefs, current)
        new_turn['belief_state'] = find_beliefs(global_beliefs, current)

        new_turn['agent_actions'] = agent_actions
        new_turn['agent_utterance'] = agent_utterance

        new_turn['num'] = {
          "agent_actions": [vectorize(aa) for aa in agent_actions],
          "utterance": vectorize(new_turn['utterance'], 'utterance')
        }

        dialogue['turns'].append(new_turn)

    revised.append(dialogue)

  shuffle(revised)  # works in place
  train_dev = int(len(revised) * 0.7)
  dev_test = int(len(revised) * 0.9)

  divided = {
    'train': revised[:train_dev],
    'val': revised[train_dev:dev_test],
    'test': revised[dev_test:]
  }

  for split in ['train', 'val', 'test']:
    data = {'dialogues': divided[split]}
    size = len(data['dialogues'])
    print(f"then we saved {size} with {split}")
    json.dump(data, open(f"{domain}/clean/{split}.json", "w"))

  print("Finished creating user intent labels, now converting ontology ...")
  half_ont = json.load( open(f'{domain}/pre_num_ontology.json', 'r') )

  full_ont = {}
  full_ont['dialogue_acts'] = half_ont['dialogue_acts']
  full_ont['slot_values'] = {}
  full_ont['vectorized'] = {}

  for dialogue in revised:
    for turn in dialogue['turns']:
      for intent in turn['user_intent']:
        if intent[0] == "task":
          pdb.set_trace()
        slotvals[intent[0]].add(intent[1])

  for slot, value_set in slotvals.items():
    empirical_values = list(value_set)
    full_ont['slot_values'][slot] = empirical_values
    numbers = [vectorize([slot, val]) for val in empirical_values]
    full_ont['vectorized'][slot] = numbers
    print("We found {} values in the {} slot".format(
        len(empirical_values), slot))

  json.dump(full_ont, open(f"{domain}/ontology.json", "w"))

import json
from collections import defaultdict
import numpy as np
from tqdm import tqdm
from stanza.nlp.corenlp import CoreNLPClient
from pprint import pprint
from objects.blocks.ontology import Ontology

client = None

def annotate(sent):
  global client
  if client is None:
    client = CoreNLPClient(default_annotators='ssplit,tokenize'.split(','))
  words = []
  for sent in client.annotate(sent).sentences:
    for tok in sent:
      words.append(tok.word)
  return words

class Intent:

  def __init__(self, domain, subdomain, act, slot, value):
    self.domain = domain        # restaurant, airline, banking
    self.subdomain = subdomain  # reservation, traffic, event
    self.act = act              # accept/reject, open/close, request/inform,
                                # question/answer, acknow/confuse
    self.slot = slot            # time, location, price, rating, name
    self.value = value          # north, yes/no, cheap, 5, more, italian

  @classmethod
  def from_dict(cls, d):
    return cls(**d)


class Turn:

  def __init__(self, turn_id, utterance, user_intent, belief_state, agent_actions, agent_utterance, num=None):
    self.id = turn_id
    self.utterance = utterance
    self.user_intent = user_intent
    self.belief_state = belief_state
    self.agent_actions = agent_actions
    self.agent_utterance = agent_utterance
    self.num = num or {}

  def to_dict(self):
    return {'turn_id': self.id, 'utterance': self.utterance, 'user_intent': self.user_intent, 'belief_state': self.belief_state, 'agent_actions': self.agent_actions, 'agent_utterance': self.agent_utterance, 'num': self.num}

  @classmethod
  def from_dict(cls, d):
    return cls(**d)

  @classmethod
  def annotate_raw(cls, raw):
    agent_actions = []
    for a in raw['agent_actions']:
      if isinstance(a, list):
        s, v = a
        agent_actions.append(['inform'] + s.split() + ['='] + v.split())
      else:
        agent_actions.append(['request'] + a.split())
    # NOTE: fix inconsistencies in data label
    fix = {'centre': 'center', 'areas': 'area', 'phone number': 'number'}
    return cls(
      turn_id=raw['turn_idx'],
      utterance=annotate(raw['utterance']),
      agent_actions=agent_actions,
      user_intent=[[fix.get(s.strip(), s.strip()), fix.get(v.strip(), v.strip())] for s, v in raw['user_intent']],
      belief_state=raw['belief_state'],
      agent_utterance=raw['agent_utterance'],
    )

  def numericalize_(self, vocab):
    self.num['utterance'] = vocab.word2index(['<sos>'] + [w.lower() for w in self.utterance + ['<eos>']], train=True)
    self.num['agent_actions'] = [vocab.word2index(['<sos>'] + [w.lower() for w in a] + ['<eos>'], train=True) for a in self.agent_actions + [['<sentinel>']]]


class Dialogue:

  def __init__(self, dialogue_id, turns):
    self.id = dialogue_id
    self.turns = turns

  def __len__(self):
    return len(self.turns)

  def to_dict(self):
    return {'dialogue_id': self.id, 'turns': [t.to_dict() for t in self.turns]}

  @classmethod
  def from_dict(cls, d):
    return cls(d['dialogue_id'], [Turn.from_dict(t) for t in d['turns']])

  @classmethod
  def annotate_raw(cls, raw):
    return cls(raw['dialogue_idx'], [Turn.annotate_raw(t) for t in raw['dialogue']])


class Dataset:

  def __init__(self, dialogues):
    self.dialogues = dialogues

  def __len__(self):
    return len(self.dialogues)

  def iter_turns(self):
    for d in self.dialogues:
      for t in d.turns:
        yield t

  def to_dict(self):
    return {'dialogues': [d.to_dict() for d in self.dialogues]}

  @classmethod
  def from_dict(cls, d):
    return cls([Dialogue.from_dict(dd) for dd in d['dialogues']])

  @classmethod
  def annotate_raw(cls, fname):
    with open(fname) as f:
      data = json.load(f)
      return cls([Dialogue.annotate_raw(d) for d in tqdm(data)])

  def numericalize_(self, vocab):
    for t in self.iter_turns():
      t.numericalize_(vocab)

  def extract_ontology(self):
    slots = set()
    values = defaultdict(set)
    for t in self.iter_turns():
      for s, v in t.user_intent:
        slots.add(s.lower())
        values[s].add(v.lower())
    return Ontology(sorted(list(slots)), {k: sorted(list(v)) for k, v in values.items()})

  def batch(self, batch_size, shuffle=False):
    turns = list(self.iter_turns())
    if shuffle:
      np.random.shuffle(turns)
    for i in tqdm(range(0, len(turns), batch_size)):
      yield turns[i:i+batch_size]

  def process_confidence(self, confidence, idx):
    for slot in ["area", "request", "price range", "food"]:
      conf = [round(x, 3) for x in confidence[slot][idx]]
      print("{} confidence: {}".format(slot, conf))

  def run_report(self, one_batch, preds, confidence):
    num_samples = len(preds)
    request = []
    inform = []
    joint_goal = []
    fix = {'centre': 'center', 'areas': 'area', 'phone number': 'number'}
    i = 0
    pred_state = {}
    for t in one_batch:
      if i >= num_samples:
        break
      gold_request = set([(s, v) for s, v in t.user_intent if s == 'request'])
      gold_inform = set([(s, v) for s, v in t.user_intent if s != 'request'])
      pred_request = set([(s, v) for s, v in preds[i] if s == 'request'])
      pred_inform = set([(s, v) for s, v in preds[i] if s != 'request'])
      request.append(gold_request == pred_request)
      inform.append(gold_inform == pred_inform)

      double_correct = (gold_request == pred_request) and (gold_inform == pred_inform)
      joint_goal.append(double_correct)

      if not double_correct:
        print(" ".join(t.utterance))
        print('actual', t.user_intent)
        print('predicted', preds[i])
        self.process_confidence(confidence, i)
        print('----------------')

      i += 1
    return {'turn_inform': np.mean(inform), 'turn_request': np.mean(request), 'joint_goal': np.mean(joint_goal)}

  def evaluate_preds(self, preds):
    request = []
    inform = []
    joint_goal = []
    fix = {'centre': 'center', 'areas': 'area', 'phone number': 'number'}
    i = 0
    for d in self.dialogues:
      pred_state = {}
      for t in d.turns:
        gold_request = set([(s, v) for s, v in t.user_intent if s == 'request'])
        gold_inform = set([(s, v) for s, v in t.user_intent if s != 'request'])
        pred_request = set([(s, v) for s, v in preds[i] if s == 'request'])
        pred_inform = set([(s, v) for s, v in preds[i] if s != 'request'])
        request.append(gold_request == pred_request)
        inform.append(gold_inform == pred_inform)

        gold_recovered = set()
        pred_recovered = set()
        for s, v in pred_inform:
          pred_state[s] = v
        for b in t.belief_state:
          for s, v in b['slots']:
            if b['act'] != 'request':
              gold_recovered.add((b['act'], fix.get(s.strip(), s.strip()),
                                            fix.get(v.strip(), v.strip())))
        for s, v in pred_state.items():
          pred_recovered.add(('inform', s, v))
        joint_goal.append(gold_recovered == pred_recovered)
        i += 1
    result = {'turn_inform': np.mean(inform),
             'turn_request': np.mean(request),
               'joint_goal': np.mean(joint_goal)}
    pprint(result)

  def record_preds(self, preds, to_file):
    data = self.to_dict()
    i = 0
    for d in data['dialogues']:
      for t in d['turns']:
        t['pred'] = sorted(list(preds[i]))
        i += 1
    with open(to_file, 'wt') as f:
      json.dump(data, f)

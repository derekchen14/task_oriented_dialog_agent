import numpy as np

class Ontology:

  def __init__(self, slots=None, values=None, num=None):
    self.slots = slots or []
    self.values = values or {}
    self.num = num or {}

  def __add__(self, another):
    new_slots = sorted(list(set(self.slots + another.slots)))
    new_values = {s: sorted(list(set(self.values.get(s, []) + another.values.get(s, [])))) for s in new_slots}
    return Ontology(new_slots, new_values)

  def __radd__(self, another):
    return self if another == 0 else self.__add__(another)

  def to_dict(self):
    return {'slots': self.slots, 'values': self.values, 'num': self.num}

  def numericalize_(self, vocab):
    self.num = {}
    for s, vs in self.values.items():
      self.num[s] = [vocab.word2index(annotate('{} = {}'.format(s, v)) + ['<eos>'], train=True) for v in vs]

  @classmethod
  def from_dict(cls, d):
    return cls(**d)
import os
import json
import numpy as np
import pickle as pkl

class Ontology:

  def __init__(self, acts=None, slots=None, relations=None, values=None, num=None):
    self.acts = acts or []
    self.slots = slots or []
    self.relations = relations or []
    self.values = values or {}
    self.num = num or {}

  def __add__(self, another):
    new_acts = sorted(list(set(self.acts + another.acts)))
    new_slots = sorted(list(set(self.slots + another.slots)))
    new_relations = sorted(list(set(self.relations + another.relations)))
    new_values = {s: sorted(list(set(
        self.values.get(s, []) + another.values.get(s, [])))) for s in new_slots}
    return Ontology(new_acts, new_slots, new_relations, new_values)

  def __radd__(self, another):
    # reflective add, which flips the ordering of adding operations
    return self if another == 0 else self.__add__(another)

  def to_dict(self):
    return {'acts': self.acts, 'relations': self.relations,
            'slots': self.slots, 'values': self.values }

  @classmethod
  def from_dict(cls, d):
    return cls(**d)

  @classmethod
  def from_path(cls, path):
    ont = json.load(open(os.path.join(path, "ontology.json"), "r"))

    data = {}
    data['acts'] = ont['dialogue_acts']
    data['slots'] = list(ont['slot_values'].keys())
    data['values'] = ont['slot_values']

    if 'relations' in ont.keys():
      data['relations'] = ont['relations']
    if 'num' in ont.keys():
      data['num'] = ont['num']

    return cls(**data)

import os
import json

class Ontology(object):
  def __init__(self, args, data_dir):
    self.args = args
    self.directory = data_dir
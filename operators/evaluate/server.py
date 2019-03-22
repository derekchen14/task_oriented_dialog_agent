#!/usr/bin/env python3

from http.server import BaseHTTPRequestHandler, HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import parse_qs, parse_qsl, urlparse
import os, re

routes = {
  "/" : "hiya World",
  "/goodbye" : "Wrap it up!",
  "/favicon.ico": "some other thing"
}

class ToyModel(object):
  def __init__(self):
    print("this is just a stub model for testing")

  def respond(self, user_input):
    return "This is a server output!"

class Handler(SimpleHTTPRequestHandler):
  def _set_headers(self):
    self.send_response(200)
    self.send_header('Access-Control-Allow-Origin', '*')
    self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    self.send_header('Content-type', 'text/html')
    self.end_headers()

  def do_GET(self):
    print("path: ", self.path)
    if self.path == "/goal":
      self._set_headers()
      new_goal = self.server.agent.get_goal()
      self.wfile.write(bytes(new_goal, "UTF-8"))
    else:
      self.path = self.server.wd+self.path
      super().do_GET()

  def do_POST(self):
    params = parse_qs(urlparse(self.path).query)
    # it's a list in case there's duplicates
    inputText = params["inputText"][0]
    raw_user_input = inputText.replace("|||","\n").strip()
    # print('raw_user_input', raw_user_input)
    user_input = self.parse_raw_input(raw_user_input)
    response = self.server.agent.respond_to_turker(user_input)
    # print('agent_response', response)
    self._set_headers()
    self.wfile.write(response.encode())

  def parse_raw_input(self, raw):
    parsed = {'inform_slots':{}, 'request_slots':{}}
    cleaned = raw.strip(' ').strip('\n').strip('\r')
    intents = cleaned.lower().split(',')
    for intent in intents:
      idx = intent.find('(')
      act = intent[0:idx]
      if re.search(r'thanks?', act):
        self.finish_episode = True
      else:
        # print("---------")
        # print(intent)
        slot, value = intent[idx+1:-1].split("=") # -1 is to skip the closing ')'
        parsed["{}_slots".format(act)][slot] = value


      parsed["dialogue_act"] = act
      parsed["nl"] = cleaned

    parsed['turn_count'] = 2
    return parsed
#!/usr/bin/env python3

from http.server import BaseHTTPRequestHandler, HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import parse_qs, parse_qsl, urlparse
import os

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
    self.path = self.server.wd+self.path
    print(self.path)
    super(Handler,self).do_GET()

  def do_POST(self):
    params = parse_qs(urlparse(self.path).query)
    # it's a list in case there's duplicates
    inputText = params["inputText"][0]
    user_input = inputText.replace("|||","\n").strip()
    print('user_input', user_input)
    response = self.server.agent.respond(user_input)
    print('agent_response', response)
    self._set_headers()
    self.wfile.write(response.encode())

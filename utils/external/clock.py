# -*- coding: utf-8 -*-
import time
import math

def asMinutes(s):
  m = math.floor(s / 60)
  s -= m * 60
  return '%dm %ds' % (m, s)

def timeSince(since, percent):
  now = time.time()
  seconds_passed = now - since
  estimated_seconds = seconds_passed / (percent)
  remaining_seconds = estimated_seconds - seconds_passed
  return '%s (est remaining: %s)' % (asMinutes(seconds_passed), asMinutes(remaining_seconds))


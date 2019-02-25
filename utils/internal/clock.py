# -*- coding: utf-8 -*-
import time as tm
import math
from tqdm import tqdm as progress_bar

def asMinutes(s):
  m = math.floor(s / 60)
  s -= m * 60
  return '%dm %ds' % (m, s)

def timeSince(since, percent):
  now = tm.time()
  seconds_passed = now - since
  estimated_seconds = seconds_passed / (percent / 100)
  remaining_seconds = estimated_seconds - seconds_passed
  return '(%s remaining)' % (asMinutes(remaining_seconds))

def time_past(since):
  now = tm.time()
  minutes_passed = (now - since)/60.0
  print('{0:.2f} minutes '.format(minutes_passed))

def starting_checkpoint(epoch, epochs, use_cuda):
  if epoch == 0:
    if use_cuda:
      print("Starting to train on GPUs on epoch {}... ".format(epoch+1))
    else:
      print("Start local CPU training on epoch {} ... ".format(epoch+1))
  else:
    print("Continuing on epoch {} of {} ...".format(epoch+1, epochs))

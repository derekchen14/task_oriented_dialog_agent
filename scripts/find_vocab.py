from nltk import word_tokenize
import json

vocab = set([])
fnames = [  "dialog-babi-task6-dstc2-dev.txt",
            "dialog-babi-task6-dstc2-tst.txt",
            "dialog-babi-task6-dstc2-trn.txt",
            "dialog-babi-task6-dstc2-candidates.txt",
            "dialog-babi-task5-full-dialogs-dev.txt",
            "dialog-babi-task5-full-dialogs-tst-OOV.txt",
            "dialog-babi-task4-phone-address-tst-OOV.txt",
            "dialog-babi-task3-options-tst-OOV.txt",
            "dialog-babi-task2-API-refine-tst-OOV.txt",
          ]

def valid(token):
  if token in ["<SILENCE>", "api_call"]:
    return False
  if "_" in token:
    return False
  return True

def pull_vocab(filename, vocab):
  with open(filename, "r") as f:
    for line in f:
      if line.strip() == "":
        continue
      nid, line = line.split(' ', 1)
      line = line.decode('utf-8').strip()
      if len(line.split('\t')) == 1:
        continue
      u, r = line.split('\t')
      for token in word_tokenize(u):
        if valid(token):
          vocab.add(token)
      for token in word_tokenize(r):
        if valid(token):
          vocab.add(token)
  print("Done with {0} now have {1} words".format(filename, len(vocab)) )
  return vocab

for filename in fnames:
  vocab = pull_vocab(filename, vocab)

vocab = list(vocab)
vocab.sort()

special_tokens = ["<T01>","<T02>","<T03>","<T04>","<T05>","<T06>","<T07>",
                  "<T08>","<T09>","<T10>","<T11>","<T12>","<T13>","<T14>",
                  "UNK", "SOS", "EOS", "api_call", "<SILENCE>", "Reserve"]
all_tokens = special_tokens + vocab

print len(all_tokens)

json.dump(all_tokens, open("vocab.json", "w"))



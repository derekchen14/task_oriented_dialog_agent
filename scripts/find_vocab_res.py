from nltk import word_tokenize
import json

vocab = set([])
fnames = [  "dialog-babi-task6-dstc2-dev.txt",
            "dialog-babi-task6-dstc2-tst.txt",
            "dialog-babi-task6-dstc2-trn.txt",
            "dialog-dstc-candidates.txt",
            "dialog-babi-task5-trn.txt",
            "dialog-babi-task4-trn.txt",
            "dialog-babi-task3-trn.txt",
            "dialog-babi-task2-trn.txt",
            "dialog-babi-task5-tst-OOV.txt",
            "dialog-babi-task4-tst-OOV.txt",
            "dialog-babi-task3-tst-OOV.txt",
            "dialog-babi-task2-tst-OOV.txt",
          ]

'''
Checks if the token is a normal, valid word
If the token is valid, then return True
Otherwise, it is a special keywork, so return False
'''
def valid(token):
  if token in ["<SILENCE>", "api_call"]:
    return False
  if "address" in token and token.startswith("resto"):
    return False
  if "phone" in token and token.startswith("resto"):
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
      # for token in u.split():
      #   if valid(token):
      #     vocab.add(token)
      # for token in r.split():
        if valid(token):
          vocab.add(token)
  print("Done with {0} now have {1} words".format(filename, len(vocab)) )
  return vocab

for filename in fnames:
  vocab = pull_vocab(filename, vocab)

vocab = list(vocab)
vocab.sort()

special_tokens = ["<SILENCE>", "<T01>","<T02>","<T03>","<T04>","<T05>","<T06>",
                  "<T07>","<T08>","<T09>","<T10>","<T11>","<T12>","<T13>",
                  "<T14>","UNK", "SOS", "EOS", "api_call","<PHONE>", "<ADDR>"]
all_tokens = special_tokens + vocab

print len(all_tokens)

json.dump(all_tokens, open("res_vocab.json", "w"))




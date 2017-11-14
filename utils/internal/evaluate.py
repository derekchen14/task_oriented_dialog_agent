# Get the first minibatch in the dev set.
minibatch = get_parallel_minibatch(
  lines=dev_lines, src_word2id=src_word2id,
  trg_word2id=trg_word2id, index=0, batch_size=batch_size,
  volatile=True
)

if cuda_available:
  minibatch['input_src'] = minibatch['input_src'].cuda()
  minibatch['input_trg'] = minibatch['input_trg'].cuda()
  minibatch['output_trg'] = minibatch['output_trg'].cuda()

# Run it through our model (in teacher forcing mode)
res = seq2seq(
  input_src=minibatch['input_src'], input_trg=minibatch['input_trg'], src_lengths=minibatch['src_lens']
)

# Pick the most likely word at each time step
res = res.data.cpu().numpy().argmax(axis=-1)

# Cast targets to numpy
gold = minibatch['output_trg'].data.cpu().numpy()

# Decode indices to words for predictions and gold
res = [[trg_id2word[x] for x in line] for line in res]
gold = [[trg_id2word[x] for x in line] for line in gold]

for r, g in zip(res, gold):
  if '</s>' in r:
    index = r.index('</s>')
  else:
    index = len(r)

  print('Prediction : %s ' % (' '.join(r[:index])))

  index = g.index('</s>')
  print('Gold : %s ' % (' '.join(g[:index])))
  print('---------------')
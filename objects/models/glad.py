import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from objects.blocks.attention import Attention
from objects.components import device

# the GlobalLocalModel model described in https://arxiv.org/abs/1805.09655.
class GLAD(nn.Module):
  def __init__(self, args, ontology, vocab, Eword, GLADEncoder):
    super().__init__(args)
    self.optimizer = None
    self.model_type = "glad"
    self.attend = Attention()

    self.vocab = vocab
    self.ontology = ontology
    if args.pretrained:
      self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(Eword))
    else:
      self.embedding = nn.Embedding(len(vocab), self.demb)  # (num_embeddings, embedding_dim)

    self.demb = args.embedding_size    # aka embedding dimension
    self.dhid = args.hidden_dim       # aka hidden state dimension
    dropout = {key: args.drop_prob for key in ["emb", "local", "global"]}

    self.utt_encoder = GLADEncoder(self.demb, self.dhid, ontology.slots, dropout)
    self.act_encoder = GLADEncoder(self.demb, self.dhid, ontology.slots, dropout)
    self.ont_encoder = GLADEncoder(self.demb, self.dhid, ontology.slots, dropout)
    self.utt_scorer = nn.Linear(2 * self.dhid, 1)
    self.score_weight = nn.Parameter(torch.Tensor([0.5]))

  def forward(self, batch):
    # convert to variables and look up embeddings
    eos = self.vocab.word2index('<eos>')
    utterance, utterance_len = self.pad([e.num['utterance'] for e in batch], self.embedding, pad=eos)
    acts = [self.pad(e.num['agent_actions'], self.embedding, pad=eos) for e in batch]
    ontology = {s: self.pad(v, self.embedding, pad=eos) for s, v in self.ontology.num.items()}

    ys = {}
    for s in self.ontology.slots:
      # for each slot, compute the scores for each value
      H_utt, c_utt = self.utt_encoder(utterance, utterance_len, slot=s)
      # H_utt: torch.Size([50, 30, 400])  batch_size x seq_len x embed_dim
      # c_utt: torch.Size([50, 400])
      _, C_acts = list(zip(*[self.act_encoder(a, a_len, slot=s) for a, a_len in acts]))
      _, C_vals = self.ont_encoder(ontology[s][0], ontology[s][1], slot=s)
      # C_acts is list of length 50, a single c_act is size([1, 400])
      # C_vals is list of length 7, a single c_val is size([400])

      # compute the utterance score
      y_utts = []
      q_utts = []
      for i, c_val in enumerate(C_vals):
        q_utt = self.attend(H_utt, c_val.unsqueeze(0).expand(
                            len(batch), *c_val.size()), lengths=utterance_len)
        q_utts.append(q_utt)   # torch.Size([50, 400])
      y_utts = self.utt_scorer(torch.stack(q_utts, dim=1)).squeeze(2)

      # compute the previous action score
      q_acts = []
      for j, C_act in enumerate(C_acts):
        q_act = self.attend(C_act.unsqueeze(0), c_utt[j].unsqueeze(0), [C_act.size(0)])
        q_acts.append(q_act)  # torch.Size([1, 400])
      # (50x7) =         (50, 400)       x    (400, 7)
      y_acts = torch.cat(q_acts, dim=0).mm(C_vals.transpose(0, 1))

      # combine the scores
      # y_acts: torch.Size([50, 7])  batch size, num of values for slot 's'
      # y_utts: torch.Size([50, 7])  for slot==area, there are 7 values
      ys[s] = torch.sigmoid(y_utts + self.score_weight * y_acts)
      # ys[s] = torch.sigmoid(c_utt.mm(C_vals.transpose(0, 1)))

    if self.training:
      # create label variable and compute loss
      labels = {s: [len(self.ontology.values[s]) * [0] for i in range(len(batch))] for s in self.ontology.slots}
      for i, e in enumerate(batch):
        for s, v in e.user_intent:
          labels[s][i][self.ontology.values[s].index(v)] = 1
      labels = {s: torch.Tensor(m).to(device) for s, m in labels.items()}

      loss = 0
      for s in self.ontology.slots:
        loss += self.criterion(ys[s], labels[s])
    else:
      loss = torch.Tensor([0]).to(device)
    return loss, {s: v.data.tolist() for s, v in ys.items()}

  def pad(self, seqs, emb, pad=0):
    lens = [len(s) for s in seqs]
    max_len = max(lens)
    padded = torch.LongTensor([s + (max_len-l) * [pad] for s, l in zip(seqs, lens)])
    return emb(padded.to(device)), lens
    # out = F.dropout(emb(padded.to(device)), drop_rate)
    # return out, lens
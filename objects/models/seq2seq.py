import random
import numpy as np
from torch import nn

class Seq2Seq(nn.Module):
  def __init__(self, encoder, decoder):
    super(Seq2Seq, self).__init__()
    self.encoder = encoder.to(device)
    self.decoder = decoder.to(device)
    self.type = "seq2seq"
    self.arguments_size = decoder.arguments_size

  def flatten_parameters(self):
    self.encoder.rnn.flatten_parameters()
    self.decoder.rnn.flatten_parameters()

  def forward(self, sources, targets, enc_hidden, enc_length, dec_length, track):
    loss, predictions, visual, teach_ratio = track
    self.flatten_parameters()
    encoder_outputs, enc_hidden = self.encoder(sources, enc_hidden)

    dec_hidden = enc_hidden
    decoder_input = var([[vocab.SOS_token]], "long")
    decoder_context = var(torch.zeros(1, 1, decoder.hidden_size))

    for di in range(dec_length):
      use_teacher_forcing = random.random() < teach_ratio
      if self.arguments_size == "large":
        dec_output, dec_context, dec_hidden, attn_weights = self.decoder(
          decoder_input, dec_context, dec_hidden, encoder_outputs,
          sources, targets, di, use_teacher_forcing)
      elif self.arguments_size == "medium":
        dec_output, dec_context, dec_hidden, attn_weights = self.decoder(
            decoder_input, dec_context, dec_hidden, encoder_outputs)
      elif self.arguments_size == "small":
        dec_output, dec_context = self.decoder(decoder_input, dec_context)
        attn_weights, visual = False, False

      # visual[:, di] = attn_weights.squeeze(0).squeeze(0).cpu().data
      loss += criterion(dec_output, targets[di])

      if use_teacher_forcing:
        decoder_input = targets[di]
      else:       # Use the predicted word as the next input
        topv, topi = dec_output.data.topk(1)
        ni = topi[0][0]
        predictions.append(ni)
        if ni == vocab.EOS_token:
          break
        decoder_input = var([[ni]], "long")

    return loss, predictions, visual
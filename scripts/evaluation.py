######################################################################
# Evaluation
# ==========
#
# Evaluation is mostly the same as training, but there are no targets so
# we simply feed the decoder's predictions back to itself for each step.
# Every time it predicts a word we add it to the output string, and if it
# predicts the EOS token we stop there. We also store the decoder's
# attention outputs for display later.
#

def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
  input_variable = variableFromSentence(input_lang, sentence)
  input_length = input_variable.size()[0]
  encoder_hidden = encoder.initHidden()

  encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
  encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

  for ei in range(input_length):
    encoder_output, encoder_hidden = encoder(input_variable[ei],
                         encoder_hidden)
    encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]

  decoder_input = Variable(torch.LongTensor([[SOS_token]]))  # SOS
  decoder_input = decoder_input.cuda() if use_cuda else decoder_input

  decoder_hidden = encoder_hidden

  decoded_words = []
  decoder_attentions = torch.zeros(max_length, max_length)

  for di in range(max_length):
    decoder_output, decoder_hidden, decoder_attention = decoder(
      decoder_input, decoder_hidden, encoder_output, encoder_outputs)
    decoder_attentions[di] = decoder_attention.data
    topv, topi = decoder_output.data.topk(1)
    ni = topi[0][0]
    if ni == EOS_token:
      decoded_words.append('<EOS>')
      break
    else:
      decoded_words.append(output_lang.index2word[ni])
    decoder_input = Variable(torch.LongTensor([[ni]]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

  return decoded_words, decoder_attentions[:di + 1]


######################################################################
# We can evaluate random sentences from the training set and print out the
# input, target, and output to make some subjective quality judgements:
#

def evaluateRandomly(encoder, decoder, n=10):
  for i in range(n):
    pair = random.choice(pairs)
    print('>', pair[0])
    print('=', pair[1])
    output_words, attentions = evaluate(encoder, decoder, pair[0])
    output_sentence = ' '.join(output_words)
    print('<', output_sentence)
    print('')

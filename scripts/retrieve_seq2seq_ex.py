'''
This script retrieves answers to questions in PATH.txt.
Pretrained models.pt can be loaded to answer the questions
'''
from nltk import word_tokenize
from utils.external.preprocessers import *
from torch.autograd import Variable

sys.path.append('.')

PATH = 'scripts/seq2seq_ex.txt'

def read_questions(path):
    '''
    :param path: path of your questions. See 'scripts/seq2seq_ex.txt' for example.
                 One question per line
    :return: list of tokenized questions
    '''
    questions = []
    with open(path) as f:
        lines = f.readlines()
        for line in lines:
            line = line.decode('utf-8').strip()
            q = word_tokenize(line)
            questions.append(q)
    return questions


def q2idx(questions):
    '''
    :param questions: list of tokens
    :return: convert tokens to index
    '''
    results = []
    for q in questions:
        question, wop = variable_from_sentence(q, [questions.index(q)+1])
        results.append(question)
    return results



def answers(input_variable, encoder, decoder):
    '''
    :param input_variable: list of index
    :param encoder: pretrained encoder
    :param decoder: pretrained decoder
    :return:
    '''
    max_length = 20

    encoder_hidden = encoder.initHidden()
    input_length = input_variable.size()[0]
    target_length = 25
    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[vocab.SOS_token]]))
    decoder_hidden = encoder_hidden


    decoder_outputs = []
    for di in range(target_length):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]

        decoder_outputs.append(ni)

        decoder_input = Variable(torch.LongTensor([[ni]]))
        if ni == vocab.EOS_token:
            break

    return decoder_outputs


questions = read_questions(PATH)
questions = q2idx(questions)

# encoder = torch.load('encoder.pt')
# decoder = torch.load('decoder.pt')

encoder = torch.load('encoder_7500t1.pt')
decoder = torch.load('decoder_7500t1.pt')

encoder.eval()
decoder.eval()

print 'Model loaded!'


for q in questions:
    a = answers(q, encoder, decoder)
    answer = []
    for i in a:
        answer.append(vocab.index_to_word(i))
    aSen = ' '.join(answer)
    print aSen




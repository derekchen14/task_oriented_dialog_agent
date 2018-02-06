from torch import optim
from torch import cuda
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm

use_cuda = cuda.is_available()

def init_optimizers(optimizer_type, enc_params, dec_params, lr, weight_decay):
  if optimizer_type == 'SGD':
    encoder_optimizer = optim.SGD(enc_params, lr, weight_decay)
    decoder_optimizer = optim.SGD(dec_params, lr, weight_decay)
  elif optimizer_type == 'Adam':
    encoder_optimizer = optim.Adam(enc_params, lr * 0.01, weight_decay)
    decoder_optimizer = optim.Adam(dec_params, lr * 0.01, weight_decay)
  else:
    encoder_optimizer = optim.RMSprop(enc_params, lr, weight_decay)
    decoder_optimizer = optim.RMSprop(dec_params, lr, weight_decay)
  return encoder_optimizer, decoder_optimizer

def smart_variable(tensor):
  result = Variable(tensor)
  if use_cuda:
    return result.cuda()
  else:
    return result

def clip_gradient(models, clip):
  '''
  models: a list, such as [encoder, decoder]
  clip: amount to clip the gradients by
  '''
  if clip is None:
    return
  for model in models:
    clip_grad_norm(model.parameters(), clip)
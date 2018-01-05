import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def plot(xs, ys, title, xlabel, ylabel):
  assert len(xs) == len(ys)

  for i in range(len(xs)):
    plt.plot(xs[i], ys[i])

  plt.title(title)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.legend(['Train', "Validation"])
  plt.show()




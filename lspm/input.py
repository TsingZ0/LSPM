import numpy as np

class DataInput:
  def __init__(self, data, batch_size, k):
    self.k = k
    self.batch_size = batch_size
    self.data = data
    self.epoch_size = len(self.data) // self.batch_size
    if self.epoch_size * self.batch_size < len(self.data):
      self.epoch_size += 1
    self.i = 0

  def __iter__(self):
    return self

  def __next__(self):

    if self.i == self.epoch_size:
      raise StopIteration

    ts = self.data[self.i * self.batch_size : min((self.i+1) * self.batch_size, len(self.data))]
    self.i += 1

    u, i, j = [], [], []
    for t in ts:
      u.append(t[0])
      i.append(t[2][0])
      j.append(t[2][1])

    hist_i = np.zeros([len(ts), self.k], np.int32)

    kk = 0
    for t in ts:
      length = len(t[1])
      if length > self.k:
        for l in range(self.k):
          hist_i[kk][l] = t[1][length-self.k+l]
      else:
        for l in range(length):
          hist_i[kk][l] = t[1][l]
      kk += 1

    return self.i, (u, i, j, hist_i)

class DataInputTest:
  def __init__(self, data, batch_size, k):
    self.k = k
    self.batch_size = batch_size
    self.data = data
    self.epoch_size = len(self.data) // self.batch_size
    if self.epoch_size * self.batch_size < len(self.data):
      self.epoch_size += 1
    self.i = 0

  def __iter__(self):
    return self

  def __next__(self):

    if self.i == self.epoch_size:
      raise StopIteration

    ts = self.data[self.i * self.batch_size : min((self.i+1) * self.batch_size, len(self.data))]
    self.i += 1

    u, i, j = [], [], []
    for t in ts:
      u.append(t[0])
      i.append(t[2][0])
      j.append(t[2][1])

    hist_i = np.zeros([len(ts), self.k], np.int32)

    kk = 0
    for t in ts:
      length = len(t[1])
      if length > self.k:
        for l in range(self.k):
          hist_i[kk][l] = t[1][length-self.k+l]
      else:
        for l in range(length):
          hist_i[kk][l] = t[1][l]
      kk += 1

    return self.i, (u, i, j, hist_i)

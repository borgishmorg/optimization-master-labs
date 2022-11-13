import numpy as np
import numba as nb
from numba.experimental import jitclass

# При помощи Numba jitclass создать класс для Тензор третьего
# порядка, реализующую все основные операции с тензорами


@jitclass([
    ('_data', nb.float64[:, :, :])
])
class Tensor3:
    def __init__(self, data):
        self._data = data

    @staticmethod
    def from_zeroes(shape):
        return Tensor3(np.zeros(shape, dtype=nb.float64))

    @property
    def shape(self):
        return self._data.shape

    @property
    def data(self):
        return self._data

    # magic methods

    def __getitem__(self, index):
        return self._data[index]

    def __setitem__(self, index, value):
        self._data[index] = value

    def __add__(self, value):
        return Tensor3(self._data + value)

    def __mul__(self, value):
        return Tensor3(self._data * value)

    def __sub__(self, value):
        return Tensor3(self._data - value)

    def __truediv__(self, value):
        return Tensor3(self._data / value)

    # other methods

    def copy(self):
        return Tensor3(self._data.copy())

    def min(self):
        return self._data.min()

    def max(self):
        return self._data.max()

    def mean(self):
        return self._data.mean()

    def std(self):
        return self._data.std()


if __name__ == '__main__':
    shape = (3, 4, 4)
    t = Tensor3.from_zeroes(shape)
    print(t.shape)
    print(t.data)

    print('-='*20+'-')

    t[0] = 6
    t[1, 2:3, :] = 2
    t[2, 1, 3] = 1
    t[2, 0, :] = t[0, 0, :]
    print(t.data)

    print('-='*20+'-')

    t2 = t.copy()
    t2 *= 2
    t2 -= 2
    t2 /= 25
    t2 -= t.data
    print(t2.data)

    print('-='*20+'-')

    print(((t2 - t2.min()) / (t2.max() - t2.min())).data)

    print('-='*20+'-')

    print(((t2 - t2.mean()) / t2.std()).data)

    print('-='*20+'-')

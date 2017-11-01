from lasagne.init import Initializer, Normal, Uniform
import numpy as np

class Glorot(Initializer):

    def __init__(self, initializer, gain=1.0, c01b=False):
        if gain == 'relu':
            gain = np.sqrt(2)

        self.initializer = initializer
        self.gain = gain
        self.c01b = c01b

    def sample(self, shape):
        if self.c01b:
            #if len(shape) != 4:
            raise RuntimeError(
                "If c01b is not supported")

        else:

            if len(shape) == 2:
                n1, n2 = shape

            elif len(shape) == 4:
                n1 = np.prod(shape[2:])
                n2 = shape[0]

            else:
                raise RuntimeError(
                    "This initializer only works with shapes of length = 2 or 4")

        std = self.gain * np.sqrt(2.0 / ((n1 + n2)))
        return self.initializer(std=std).sample(shape)


class GlorotNormal(Glorot):
    """Glorot with weights sampled from the Normal distribution.
    See :class:`Glorot` for a description of the parameters.
    """
    def __init__(self, gain=1.0, c01b=False):
        super(GlorotNormal, self).__init__(Normal, gain, c01b)

class GlorotUniform(Glorot):
    """Glorot with weights sampled from the Uniform distribution.
    See :class:`Glorot` for a description of the parameters.
    """
    def __init__(self, gain=1.0, c01b=False):
        super(GlorotUniform, self).__init__(Uniform, gain, c01b)
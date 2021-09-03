try:
    import numpy
except:
    __import__('os').system('pip install numpy')

from . import core , nn, optim

__all__ = ['core','nn','optim']
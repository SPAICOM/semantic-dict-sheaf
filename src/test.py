import torch
import numpy as np

if __name__ == '__main__':
    x = np.ones((2, 3))
    print(isinstance(x, torch.Tensor))

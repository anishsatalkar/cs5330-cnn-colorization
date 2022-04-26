import torch
from torch import nn

from model import GrayscaleToColorModel


def main():
    model = GrayscaleToColorModel()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    

if __name__ == '__main__':
    main()

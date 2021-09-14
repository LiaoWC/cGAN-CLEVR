from torch import nn
import torch


# Discriminator
class Discriminator(nn.Module):
    def __init__(self, inputSize, hiddenSize, n_classes):
        super(Discriminator, self).__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(inputSize + n_classes, hiddenSize, (4, 4), (2, 2), 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hiddenSize, hiddenSize * 2, (4, 4), (2, 2), 1, bias=False),
            nn.BatchNorm2d(hiddenSize * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hiddenSize * 2, hiddenSize * 4, (4, 4), (2, 2), 1, bias=False),
            nn.BatchNorm2d(hiddenSize * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hiddenSize * 4, hiddenSize * 8, (4, 4), (2, 2), 1, bias=False),
            nn.BatchNorm2d(hiddenSize * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hiddenSize * 8, 1, (4, 4), (1, 1), 0, bias=False),  #
            nn.Sigmoid())

    def forward(self, x, label_tensor):
        # x [N,L,img_siz,img_siz]
        # label_tensor [N,L]
        # batch_siz, label_siz = label_tensor.shape[0], label_tensor.shape[1]
        img_height, img_width = x.shape[2], x.shape[3]
        label_tensor = label_tensor.view(label_tensor.shape[0], label_tensor.shape[1], 1, 1).repeat(1, 1, img_height,
                                                                                                    img_width)
        y = torch.cat((x, label_tensor), dim=1)
        y = self.trunk(y)
        return y.view(y.shape[0], y.shape[1])


# Generator
class Generator(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize, n_classes):
        super(Generator, self).__init__()
        self.hidden_size = hiddenSize
        self.fc = nn.Linear(inputSize + n_classes, hiddenSize, bias=True)
        self.main = nn.Sequential(
            nn.ConvTranspose2d(hiddenSize, hiddenSize * 8, (4, 4), (1, 1), 0, bias=False),
            nn.BatchNorm2d(hiddenSize * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(hiddenSize * 8, hiddenSize * 4, (4, 4), (2, 2), 1, bias=False),
            nn.BatchNorm2d(hiddenSize * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(hiddenSize * 4, hiddenSize * 2, (4, 4), (2, 2), 1, bias=False),
            nn.BatchNorm2d(hiddenSize * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(hiddenSize * 2, hiddenSize, (4, 4), (2, 2), 1, bias=False),
            nn.BatchNorm2d(hiddenSize),
            nn.ReLU(True),

            nn.ConvTranspose2d(hiddenSize, outputSize, (4, 4), (2, 2), 1, bias=False),
            nn.Tanh())
        # outputSize x 64 x 64

    def forward(self, noise_tensor, label_tensor):
        # "noise" [batch_size, latent_size]
        # "label" [batch_size, label_size]
        bs = noise_tensor.shape[0]
        x = torch.cat((noise_tensor, label_tensor), dim=1)
        x = self.fc(x).view(bs, self.hidden_size, 1, 1)
        return self.main(x)

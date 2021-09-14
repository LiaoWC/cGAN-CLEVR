from torch import nn
import torch


# Discriminator
class DiscriminatorNew(nn.Module):
    CLASS_PROJECT_SIZ = 128

    def __init__(self, inputSize, hiddenSize, n_classes):
        super(DiscriminatorNew, self).__init__()
        self.class_project_fc = nn.Sequential(
            nn.Linear(n_classes, 64 * 64),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.trunk = nn.Sequential(
            nn.Conv2d(inputSize + 1, hiddenSize, (4, 4), (2, 2), 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout(p=0.15),

            nn.Conv2d(hiddenSize, hiddenSize * 2, (4, 4), (2, 2), 1, bias=False),
            nn.BatchNorm2d(hiddenSize * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout(p=0.15),

            nn.Conv2d(hiddenSize * 2, hiddenSize * 4, (4, 4), (2, 2), 1, bias=False),
            nn.BatchNorm2d(hiddenSize * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout(p=0.15),

            nn.Conv2d(hiddenSize * 4, hiddenSize * 8, (4, 4), (2, 2), 1, bias=False),
            nn.BatchNorm2d(hiddenSize * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout(p=0.15),

            nn.Conv2d(hiddenSize * 8, 1, (4, 4), (1, 1), 0, bias=False),  #
            nn.Sigmoid())

    def forward(self, x, label_tensor):
        # x [N,L,img_siz,img_siz]
        # label_tensor [N,L]
        # batch_siz, label_siz = label_tensor.shape[0], label_tensor.shape[1]
        batch_size, img_height, img_width = x.shape[0], x.shape[2], x.shape[3]
        # label_tensor = label_tensor.view(label_tensor.shape[0], label_tensor.shape[1], 1, 1).repeat(1, 1, img_height,
        #                                                                                             img_width)
        label_tensor = self.class_project_fc(label_tensor).view(batch_size, 1, 64, 64)
        y = torch.cat((x, label_tensor), dim=1)
        y = self.trunk(y)
        return y.view(y.shape[0], y.shape[1])


# Generator
class GeneratorNew(nn.Module):
    CLASS_PROJECT_SIZ = 128

    def __init__(self, inputSize, hiddenSize, outputSize, n_classes):
        super(GeneratorNew, self).__init__()
        self.hidden_size = hiddenSize
        self.class_project_fc = nn.Sequential(
            nn.Linear(n_classes, self.CLASS_PROJECT_SIZ),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.fc = nn.Sequential(
            nn.Linear(inputSize + self.CLASS_PROJECT_SIZ, hiddenSize, bias=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
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
        class_project_tensor = self.class_project_fc(label_tensor)
        x = torch.cat((noise_tensor, class_project_tensor), dim=1)
        x = self.fc(x).view(bs, self.hidden_size, 1, 1)
        return self.main(x)

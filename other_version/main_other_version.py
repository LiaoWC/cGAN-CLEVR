import sys

sys.path.append('/home/ubuntu/lab5')
from model import Generator, Discriminator
import torch.nn as nn
import random
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import json
from evaluator import EvaluationModel
from PIL import Image
from typing import List
from datetime import datetime

#
CLASSES_FILE = '/home/ubuntu/lab5/objects.json'
TRAIN_FILENAME_TO_LABELS_FILE = '/home/ubuntu/lab5/train.json'
IMG_DIR = '/home/ubuntu/lab5/clever_data/images'

# Fixed settings
N_CLASSES = 24

# CUDA
os.chdir('/home/ubuntu/lab5')
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('GPU State:', DEVICE)

# Random seed
manualSeed = 7777
print('Random Seed:', manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)


def read_json(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)


CLASS_NAME_TO_NO: dict = read_json(CLASSES_FILE)
CLASS_NO_TO_NAME: dict = {CLASS_NAME_TO_NO[class_name]: class_name for class_name in CLASS_NAME_TO_NO}
TRAIN_IMG_FILENAME_TO_LABEL_NAMES: dict = read_json(TRAIN_FILENAME_TO_LABELS_FILE)


def read_img(path: str) -> np.ndarray:
    """Return (c, h, w) RGB image numpy array. """
    return np.array(Image.open(path).convert('RGB')).transpose(2, 0, 1)


def label_numbers_to_one_hot(numbers: List[int]) -> torch.Tensor:
    vec = torch.zeros(N_CLASSES)
    for number in numbers:
        vec[number] = 1.
    return vec


def label_names_to_one_hot(names: List[str]) -> torch.Tensor:
    return label_numbers_to_one_hot([CLASS_NAME_TO_NO[name] for name in names])


def get_img_label_one_hot(filename: str) -> torch.Tensor:
    return label_names_to_one_hot(TRAIN_IMG_FILENAME_TO_LABEL_NAMES[filename])


def get_example_class_labels(n) -> torch.Tensor:
    tmp = read_json(TRAIN_FILENAME_TO_LABELS_FILE)
    return torch.tensor(np.array([label_names_to_one_hot(labels).numpy() for labels in [tmp[key] for key in tmp][:n]]))


# Attributes
dataroot = '/home/ubuntu/lab5/clever_data'

BATCH_SIZ = 256
IMG_SIZ = 64
G_out_D_in = 3
G_in = 100
G_hidden = 64
D_hidden = 64

epochs = 30
lr = 0.002
beta1 = 0.5


# Weights
def weights_init(m):
    classname = m.__class__.__name__
    print('classname:', classname)

    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def show_images(images):  # TODO: add reference address
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
    plt.figure(figsize=(12, 16))
    for index, image in enumerate(images):
        plt.subplot(sqrtn, sqrtn, index + 1)
        plt.imshow(image.transpose(1, 2, 0))


class CleverDataset(Dataset):
    def __init__(self, img_dir: str):
        super().__init__()
        self.img_dir = img_dir
        # Get training image data
        train_file_content = read_json(TRAIN_FILENAME_TO_LABELS_FILE)
        filenames = []
        for root, _, files in os.walk(img_dir):
            for file in files:
                filenames.append(file)
        self.train_data = []
        for key in train_file_content:  # Check if the file exists
            if key in filenames:
                self.train_data.append({'name': key, 'label': train_file_content[key]})
        # Transform
        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZ, IMG_SIZ)),
            # transforms.CenterCrop(IMG_SIZ),
            # transforms.RandomRotation(degrees=20),
            # transforms.RandomHorizontalFlip(p=0.1),
            # transforms.Resize(IMG_SIZ),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        img_arr = Image.open(os.path.join(self.img_dir, self.train_data[idx]['name'])).convert('RGB')
        img_label = get_img_label_one_hot(self.train_data[idx]['name'])
        img_tensor = self.transform(img_arr)
        return img_tensor, img_label


INV_NORMALIZE = transforms.Normalize((-1, -1, -1), (2, 2, 2))
DATASET = CleverDataset(img_dir=IMG_DIR)
DATALOADER = torch.utils.data.DataLoader(dataset=DATASET, batch_size=BATCH_SIZ,
                                         shuffle=True)

# Create the generator
netG = Generator(G_in, G_hidden, G_out_D_in, N_CLASSES).to(DEVICE)
netG.apply(weights_init)
print(netG)

# Create the discriminator
netD = Discriminator(G_out_D_in, D_hidden, N_CLASSES).to(DEVICE)
netD.apply(weights_init)
print(netD)

# Loss fuG_out_D_intion
criterion = nn.BCELoss()
fixed_noise = torch.randn(64, G_in, device=DEVICE)
EXAMPLE_CLASS_LABELS = get_example_class_labels(64).to(DEVICE)
D_REAL_CLASS_LOSS_WEIGHT = 0  # 0.5
D_FAKE_CLASS_LOSS_WEIGHT = 0  # 0.3
G_CLASS_LOSS_WEIGHT = 0  # 0.6
REAL_LABEL = 1
FAKE_LABEL = 0

#
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

#
print('Start!')
img_list = []
G_losses = []
D_losses = []
start_time = time.time()
iters = 0

for epoch in range(epochs):
    for i, (img, class_label) in enumerate(DATALOADER, 0):
        ###################
        # --- Update D ---
        ###################
        # (1) Real data
        netD.zero_grad()
        real_cpu = img.to(DEVICE)
        class_label = class_label.to(DEVICE)
        b_size = real_cpu.size(0)
        authenticity_label = torch.full((b_size,), REAL_LABEL, device=DEVICE, dtype=torch.float)
        authenticity_output, class_output = netD(real_cpu)
        errD_real_authenticity = criterion(authenticity_output.view(-1), authenticity_label)
        errD_real_class = criterion(class_output, class_label)
        errD_real = errD_real_authenticity + errD_real_class * D_REAL_CLASS_LOSS_WEIGHT
        errD_real.backward()
        D_x = authenticity_output.mean().item()
        # (2) Fake data
        noise = torch.randn(b_size, G_in, device=DEVICE)
        fake = netG(noise, class_label)
        authenticity_label.fill_(FAKE_LABEL)
        authenticity_output, class_output = netD(fake.detach())
        errD_fake_authenticity = criterion(authenticity_output.view(-1), authenticity_label)
        errD_fake_class = criterion(class_output, class_label)
        errD_fake = errD_fake_authenticity + errD_fake_class * D_FAKE_CLASS_LOSS_WEIGHT
        errD_fake.backward()
        D_G_z1 = authenticity_output.mean().item()
        errD = errD_real + errD_fake
        errD_authenticity = errD_real_authenticity + errD_fake_authenticity
        errD_class = (errD_real_class * D_REAL_CLASS_LOSS_WEIGHT + errD_fake_class * D_REAL_CLASS_LOSS_WEIGHT)
        optimizerD.step()

        ###################
        # --- Update G ---
        ###################
        netG.zero_grad()
        authenticity_label.fill_(REAL_LABEL)
        authenticity_output, class_output = netD(fake)
        errG_authenticity = criterion(authenticity_output.view(-1), authenticity_label)
        errG_class = criterion(class_output, class_label) * G_CLASS_LOSS_WEIGHT
        errG = errG_authenticity + errG_class
        errG.backward()
        D_G_z2 = authenticity_output.mean().item()
        optimizerG.step()

        # Output training stats
        # if iters % 10 == 0:
        #     print(
        #         '[%d/%d][%d/%d](%.2f sec)\tLoss_D: %.4f(aut:%.3f,cla:%.3f)\tLoss_G: %.4f (aut:%.3f,cla:%.3f)\tD(x): %.4f D(G(z)): %.4f/%.4f' % (
        #             epoch, epochs, i, len(DATALOADER), time.time() - start_time, errD.item(), errD_authenticity.item(),
        #             errD_class.item(), errG.item(), errG_authenticity.item(), errG_class.item(), D_x, D_G_z1,
        #             D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (i % 5 == 0) or ((epoch == epochs - 1) and (i == len(DATALOADER) - 1)):
            prefix = datetime.now().strftime('%m-%d-UTC%H-%M-%S')
            print(
                '%s [%d/%d][%d/%d](%.2f sec)\tLoss_D: %.4f(aut:%.3f,cla:%.3f)\tLoss_G: %.4f (aut:%.3f,cla:%.3f)\tD(x): %.4f D(G(z)): %.4f/%.4f' % (
                    prefix, epoch, epochs, i, len(DATALOADER), time.time() - start_time, errD.item(),
                    errD_authenticity.item(),
                    errD_class.item(), errG.item(), errG_authenticity.item(), errG_class.item(), D_x, D_G_z1,
                    D_G_z2))
            with torch.no_grad():
                fake = netG(fixed_noise, EXAMPLE_CLASS_LABELS).detach().cpu()

            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
            # Plot
            fake = (fake + 1.0) / 2.0
            # fake = INV_NORMALIZE((fake+1.0)/2.0)
            # fake = INV_NORMALIZE(fake)
            # imgs_numpy = (fake.data.cpu().numpy() + 1.0) / 2.0
            imgs_numpy = fake.data.cpu().numpy()
            show_images(imgs_numpy[:64])
            plt.savefig('/home/ubuntu/lab5/plot2/' + prefix + '.png')
            plt.show()
            # Save models
            torch.save(netD, '/home/ubuntu/lab5/saved_model2/' + prefix + '-netD.pt')
            torch.save(netG, '/home/ubuntu/lab5/saved_model2/' + prefix + '-netG.pt')
            torch.save(D_losses, '/home/ubuntu/lab5/saved_model2/' + prefix + '-D_losses.pt')
            torch.save(G_losses, '/home/ubuntu/lab5/saved_model2/' + prefix + '-G_losses.pt')

        iters += 1


# Plot
def plotImage(G_losses, D_losses):
    print('Start to plot!!')
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # Grab a batch of real images from the dataloader
    real_batch = next(iter(dataLoader))

    # Plot the real images
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(
        np.transpose(vutils.make_grid(real_batch[0].to(DEVICE)[:64], padding=5, normalize=True).cpu(), (1, 2, 0)))

    # Plot the fake images from the last epoch
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
    plt.show()

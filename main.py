import sys
from model import Generator, Discriminator
from model_dropout import GeneratorNew, DiscriminatorNew
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
IMG_DIR = '/home/ubuntu/lab5/clevr_data/images'

# Fixed settings
N_CLASSES = 24

# CUDA
# os.chdir('/home/ubuntu/lab5')
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('GPU State:', DEVICE)

# Random seed
manualSeed = 6666
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


class clevrDataset(Dataset):
    def __init__(self, img_dir: str, img_siz: int):
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
            transforms.Resize((img_siz, img_siz)),
            # transforms.CenterCrop(img_siz),
            # transforms.RandomRotation(degrees=20),
            # transforms.RandomHorizontalFlip(p=0.1),
            # transforms.Resize(img_siz),
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


EM: EvaluationModel = EvaluationModel()
TEST_CASES = read_json('/home/ubuntu/lab5/test.json')
TEST_TENSORS = [label_names_to_one_hot(case) for case in TEST_CASES]


def evaluate(net_g, only_plot=False):
    score_sum = 0
    n_files = 0
    all_synth_image = torch.tensor([], device=DEVICE)
    test_fixed_noise = torch.zeros(1, G_in, device=DEVICE)
    for test_tensor in TEST_TENSORS:
        test_tensor = test_tensor.view(1, -1).to(DEVICE)
        synth_img = net_g(test_fixed_noise, test_tensor)
        if not only_plot:
            score = EM.eval(synth_img, test_tensor)
            score_sum += score
            n_files += 1
        if all_synth_image.shape[0] < 64:
            all_synth_image = torch.cat((all_synth_image, synth_img), dim=0)
    if not only_plot:
        avg_score = score_sum / n_files
        print('Testing avg score:', avg_score)
        prefix_ = datetime.now().strftime('%m-%d-UTC%H-%M-%S') + f'-AvgScore{avg_score}'
        vutils.save_image(vutils.make_grid(INV_NORMALIZE(all_synth_image)),
                          '/home/ubuntu/lab5/make_grids/' + prefix_ + '-InvNorm.png')
        return avg_score
    else:
        plt.imshow(vutils.make_grid(INV_NORMALIZE(all_synth_image)).cpu().numpy().transpose(1, 2, 0))
        plt.axis('off')
        plt.show()


def new_test_evaluate(net_g, only_plot=False):
    NEW_TEST_CASES = [["red cube"], ["yellow cylinder"], ["cyan cylinder"], ["red sphere"],
                      ["brown sphere", "yellow sphere"],
                      ["gray cube", "yellow cube"], ["brown cylinder", "cyan cylinder"],
                      ["gray cylinder", "yellow sphere"],
                      ["cyan sphere", "yellow sphere"], ["brown cylinder", "green sphere"],
                      ["brown cylinder", "gray sphere"],
                      ["blue cube", "red sphere"], ["red sphere", "yellow sphere"], ["blue cylinder", "green sphere"],
                      ["brown sphere", "gray cylinder"], ["cyan cube", "gray cube"],
                      ["blue cylinder", "blue sphere", "green cylinder"],
                      ["cyan cylinder", "purple sphere", "red cylinder"],
                      ["blue cylinder", "gray sphere", "purple cylinder"],
                      ["blue cylinder", "gray cylinder", "red sphere"],
                      ["brown cylinder", "green cylinder", "purple sphere"],
                      ["blue cylinder", "cyan cylinder", "purple cylinder"],
                      ["cyan cylinder", "gray sphere", "red cylinder"], ["cyan sphere", "purple cube", "yellow cube"],
                      ["blue cube", "green cylinder", "yellow sphere"], ["blue cube", "brown sphere", "red cylinder"],
                      ["green cube", "red cube", "yellow cylinder"], ["brown cylinder", "red sphere", "yellow cube"],
                      ["brown cylinder", "gray cylinder", "red cylinder"], ["brown cube", "gray cube", "yellow cube"],
                      ["brown sphere", "gray cylinder", "gray sphere"], ["cyan sphere", "green cube", "green cylinder"]]
    NEW_TEST_TENSORS = [label_names_to_one_hot(case) for case in NEW_TEST_CASES]
    score_sum = 0
    n_files = 0
    all_synth_image = torch.tensor([], device=DEVICE)
    test_fixed_noise = torch.zeros(1, G_in, device=DEVICE)
    for test_tensor in NEW_TEST_TENSORS:
        test_tensor = test_tensor.view(1, -1).to(DEVICE)
        synth_img = net_g(test_fixed_noise, test_tensor)
        if not only_plot:
            score = EM.eval(synth_img, test_tensor)
            score_sum += score
            n_files += 1
        if all_synth_image.shape[0] < 64:
            all_synth_image = torch.cat((all_synth_image, synth_img), dim=0)
    if not only_plot:
        avg_score = score_sum / n_files
        print('Testing avg score:', avg_score)
        prefix_ = datetime.now().strftime('%m-%d-UTC%H-%M-%S') + f'-AvgScore{avg_score}'
        vutils.save_image(vutils.make_grid(INV_NORMALIZE(all_synth_image)),
                          '/home/ubuntu/lab5/make_grids/' + prefix_ + '-InvNorm.png')
        return avg_score
    else:
        plt.imshow(vutils.make_grid(INV_NORMALIZE(all_synth_image)).cpu().numpy().transpose(1, 2, 0))
        plt.axis('off')
        plt.show()


# Attributes
BATCH_SIZ = 1024
IMG_SIZ = 64
G_out_D_in = 3
G_in = 100  # 100
G_hidden = 64  # 64
D_hidden = 64  # 64

epochs = 100
lr = 0.001
beta1 = 0.5

INV_NORMALIZE = transforms.Normalize((-1, -1, -1), (2, 2, 2))
DATASET = clevrDataset(img_dir=IMG_DIR, img_siz=IMG_SIZ)
DATALOADER = torch.utils.data.DataLoader(dataset=DATASET, batch_size=BATCH_SIZ,
                                         shuffle=True)

# Create the generator
netG = GeneratorNew(G_in, G_hidden, G_out_D_in, N_CLASSES).to(DEVICE)
netG.apply(weights_init)
print(netG)

# Create the discriminator
# netD = Discriminator(G_out_D_in, D_hidden, N_CLASSES).to(DEVICE)
netD = DiscriminatorNew(G_out_D_in, D_hidden, N_CLASSES).to(DEVICE)
netD.apply(weights_init)
# netG = torch.load('/home/ubuntu/lab5/saved_model2/08-22-UTC08-44-05-netG-0.7135416666666667.pt')
# netD = torch.load('/home/ubuntu/lab5/saved_model2/08-22-UTC08-44-05-netD-0.7135416666666667.pt')
print(netD)

# Loss fuG_out_D_init
criterion = nn.BCELoss()
# fixed_noise = torch.randn(64, G_in, device=DEVICE)
# EXAMPLE_CLASS_LABELS = get_example_class_labels(64).to(DEVICE)
REAL_LABEL = 1
FAKE_LABEL = 0
LOSS_D_REAL_WEIGHT = 1.0
LOSS_D_FAKE_WEIGHT = 1.0  # 1.0  # 0.5

N_DISCRIMINATOR_TIMES = 1
N_GENERATOR_TIMES = 1

#
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

#
print('Start!')
# img_list = []
# G_losses = torch.load('/home/ubuntu/lab5/saved_model2/08-22-UTC08-44-05-netG-0.7135416666666667.pt')
# D_losses = torch.load('/home/ubuntu/lab5/saved_model2/08-22-UTC08-10-45-D_losses-0.5468749999999999.pt')
G_losses = []
D_losses = []
model_test_scores = []
start_time = time.time()
iters = len(G_losses)  # 0
cur_max_test_avg_score = 0.0

# Train
for epoch in range(epochs):
    for i, (img, class_label) in enumerate(DATALOADER, 0):
        ###################
        # --- Update D ---
        ###################
        # (1) Real data
        b_size = img.size(0)
        for _ in range(N_DISCRIMINATOR_TIMES):
            netD.zero_grad()
            real_cpu = img.to(DEVICE)
            class_label = class_label.to(DEVICE)
            label = torch.full((b_size,), REAL_LABEL, device=DEVICE, dtype=torch.float)
            output = netD(real_cpu, class_label)
            errD_real = criterion(output.view(-1), label) * LOSS_D_REAL_WEIGHT
            errD_real.backward()
            D_x = output.mean().item()
            # (2) Fake data
            noise = torch.randn(b_size, G_in, device=DEVICE)
            fake = netG(noise, class_label)
            label.fill_(FAKE_LABEL)
            output = netD(fake.detach(), class_label)
            errD_fake = criterion(output.view(-1), label) * LOSS_D_FAKE_WEIGHT
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

        ###################
        # --- Update G ---
        ###################
        for i_generator_time in range(N_GENERATOR_TIMES):
            netG.zero_grad()
            label = torch.full((b_size,), REAL_LABEL, device=DEVICE, dtype=torch.float)  # label.fill_(REAL_LABEL)
            if i_generator_time > 0:
                noise = torch.randn(b_size, G_in, device=DEVICE)
                fake = netG(noise, class_label)
            output = netD(fake, class_label)
            errG = criterion(output.view(-1), label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if iters % 6 == 0:  # or ((epoch == epochs - 1) and (i == len(DATALOADER) - 1)):
            prefix = '/home/ubuntu/lab5/saved_model0914/' + datetime.now().strftime('%m-%d-UTC%H-%M-%S')
            print(
                '%s [%d/%d][%d/%d](%.2f sec)\tLoss_D: %.4f\tLoss_G: %.4f \tD(x): %.4f D(G(z)): %.4f/%.4f' % (
                    prefix, epoch, epochs, i, len(DATALOADER), time.time() - start_time, errD.item(),
                    errG.item(), D_x, D_G_z1, D_G_z2))
            avg_s = evaluate(netG, only_plot=False)
            model_test_scores.append(avg_s)

            # Save models
            if avg_s > cur_max_test_avg_score:
                torch.save(netD, prefix + f'-netD-{avg_s}.pt')
                torch.save(netG, prefix + f'-netG-{avg_s}.pt')
                cur_max_test_avg_score = avg_s
                print('== New Best Score! ==')

        iters += 1
    evaluate(netG, only_plot=True)


# Plot
def plotImage(G_losses_, D_losses_, test_score_, test_score_iter_freq: int, img_title=None):
    print('Start to plot!!')
    fig, ax1 = plt.subplots(figsize=(10, 7))
    ax2 = ax1.twinx()

    ax1.plot(G_losses_, label="Generator Loss", color='C0')
    ax1.plot(D_losses_, label="Discriminator Loss", color='C1')
    ax1.set_ylabel('Loss')
    ax1.set_xlabel("Iterations")
    ax1.legend()

    ax2.plot((np.arange(len(test_score_)) + 1) * test_score_iter_freq, test_score_,
             label=f'Testing Score (every {test_score_iter_freq} iter)', color='C2')
    ax2.set_ylabel('Score')
    ax2.set_ylim([0, 1.0])
    ax2.legend()

    plt.title(img_title if img_title else "Generator and Discriminator Loss During Training")
    plt.show()

# ====== Example ======
# G_losses = torch.load('/home/ubuntu/lab5/saved_model2/08-22-UTC14-23-06-G_losses-0.6041666666666666.pt')
# D_losses = torch.load('/home/ubuntu/lab5/saved_model2/08-22-UTC14-23-06-D_losses-0.6041666666666666.pt')
# plotImage(G_losses, D_losses, model_test_scores, test_score_iter_freq=6,
#           img_title=f'G_hidden={G_hidden} D_hidden={D_hidden}')
#
# # test_net_g = torch.load('/home/ubuntu/lab5/saved_model2/08-22-UTC08-44-05-netG-0.7135416666666667.pt')
# # test_net_g = torch.load('/home/ubuntu/lab5/saved_model4/08-23-UTC10-29-55-netG-0.6979166666666666.pt')
# test_net_g = torch.load('/home/ubuntu/lab5/saved_model5/08-23-UTC20-13-41-netG-0.703125.pt')
# print('------ test.json ------')
# evaluate(test_net_g, only_plot=False)
# # evaluate(test_net_g, only_plot=True)
# print('------ new_test.json ------')
# new_test_evaluate(test_net_g, only_plot=False)
#
# em: EvaluationModel = EvaluationModel()
# test_cases = read_json('/home/ubuntu/lab5/test.json')
# test_tensors = [label_names_to_one_hot(case) for case in test_cases]
# score_sum = 0
# n_files = 0
# all_synth_image = torch.tensor([], device=DEVICE)
# # test_fixed_noise = torch.load('/home/ubuntu/lab5/test_noise.pt')  # torch.randn(1, G_in, device=DEVICE)
# # test_fixed_noise = torch.randn(1, G_in, device=DEVICE)
# test_fixed_noise = torch.zeros(1, G_in, device=DEVICE)
# # torch.save(test_fixed_noise,'/home/ubuntu/lab5/test_noise.pt')
# for test_tensor in test_tensors:
#     test_tensor = test_tensor.view(1, -1).to(DEVICE)
#     synth_img = netG(test_fixed_noise, test_tensor)
#     score = em.eval(synth_img, test_tensor)
#     score_sum += score
#     n_files += 1
#     all_synth_image = torch.cat((all_synth_image, synth_img), dim=0)
# avg_score = score_sum / n_files
# print(avg_score)

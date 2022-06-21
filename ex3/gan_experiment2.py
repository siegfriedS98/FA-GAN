import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import copy

import itertools
import matplotlib.pyplot as plt

import os

from torchvision.datasets import VisionDataset

from ex3.Generator import Generator
from ex3.DiscriminatorSimple import DiscriminatorSimple
from ex3.GeneratorSimple import GeneratorSimple
from ex3.Discriminator import Discriminator
from ex3.GAN import GAN
import ex3.util as utils
from ex3.constants import USE_SIGMOID
from biotorch.module.biomodule import BioModule


def run_experiment(dataloader, plot, num_epochs, dims, dims_total):
    print("starting experiment...")

    # Loading models
    device = torch.device("cpu")

    # Create discriminator and generator
    """mode = 10"""
    mode = 0

    generator = Generator(in_features=100, out_features=dims_total, mode=mode, sig=False)
    discriminator = Discriminator(in_features=dims_total, out_features=2, mode=mode)
    gan = BioModule(GAN(in_features_g=100, out_features_g=dims_total, in_features_d=dims_total, out_features_d=2,
                        mode=0, discriminator=discriminator, generator=generator),
                    mode="fa", output_dim=2)

    # Create 100 test_noise for visualizing how well our model perform.
    test_noise = utils.noise(100).to(device)

    # Optimizers and loss
    lr_d = 0.0002
    lr_g = 0.0002
    optimizerD = optim.Adam(gan.module.discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))
    optimizerG = optim.Adam(gan.module.generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
    criterion = nn.BCELoss()



    """alignment = gan.module.generator.hidden_2[0].compute_alignment()
    print(alignment)
    alignment = gan.module.generator.hidden_1[0].compute_alignment()
    print(alignment)
    alignment = gan.module.generator.hidden_0[0].compute_alignment()
    print(alignment)
    alignment = gan.module.discriminator.hidden_2[0].compute_alignment()
    print(alignment)
    alignment = gan.module.discriminator.hidden_1[0].compute_alignment()
    print(alignment)
    alignment = gan.module.discriminator.hidden_0[0].compute_alignment()
    print(alignment)"""

    # True and False Labels.  128 is the batch size
    true_label = torch.ones(128, 2).to(device)
    false_label = torch.zeros(128, 2).to(device)

    # Create folder to hold result
    result_folder = 'gan3-result-sigmoid' if USE_SIGMOID else 'gan3-result'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    print("Starting Training...")

    discriminator_loss_history = []
    generator_loss_history = []



    for epoch in range(1, num_epochs + 1):
        discriminator_batch_loss = 0.0
        generator_batch_loss = 0.0
        for batch_idx, (data, targets) in enumerate(dataloader):
            data = data.to(device)
            batch_size = data.shape[0]
            real_label = targets.to(device)
            old_parameters = {}
            for name, param in gan.module.generator.named_parameters():
                old_parameters[name] = copy.deepcopy(param)
            gan.module.discriminator.zero_grad()

            # Train discriminator to get better at differentiate real/fake data
            # 1.1 Train discriminator on real data
            d_real_predict = gan(data.view(data.shape[0], -1), true_label, criterion)
            d_real_loss = criterion(d_real_predict, true_label).to(device)
            d_real_loss.backward()

            # 1.2 Train discriminator on fake data from generator
            d_fake_noise = utils.noise(batch_size).to(device)
            # Generate outputs and detach to avoid training the Generator on these labels

            d_fake_predict = gan(d_fake_noise, false_label, criterion)
            d_fake_loss = criterion(d_fake_predict, false_label)

            d_fake_loss.backward()


            # 1.3 combine real loss and fake loss for discriminator
            discriminator_loss = d_real_loss.to(device) + d_fake_loss
            discriminator_batch_loss += discriminator_loss.item()
            optimizerD.step()

            # Train generator to get better at deceiving discriminator
            g_fake_noise = utils.noise(batch_size).to(device)

            gan.module.generator.zero_grad()
            g_fake_predict = gan(g_fake_noise, true_label, criterion)
            generator_loss = criterion(g_fake_predict, true_label)
            generator_batch_loss += generator_loss.item()
            generator_loss.backward()
            optimizerG.step()

            """changes = []
            for name, param in gan.module.generator.named_parameters():
                if 'weight' in name and 'backward' not in name:
                    param_old = old_parameters[name]
                    param_old = param_old.detach().numpy().flatten()
                    param = param.detach().numpy().flatten()
                    changes = param_old - param
                    for co in range(len(param)):
                        changes.append(abs(param[co]-param_old[co]))
            plt.hist(changes)
            plt.show()"""

            # print loss every 100 batches
            if (batch_idx + 1) % 100 == 0:
                print(f'Epoch [{epoch}/{num_epochs}]  Batch {batch_idx + 1}/{len(dataloader)} \
                            Loss D: {discriminator_loss:.4f}, Loss G: {generator_loss:.4f}')
                """alignment = gan.module.generator.hidden_2[0].compute_alignment()
                print(alignment)
                alignment = gan.module.generator.hidden_1[0].compute_alignment()
                print(alignment)
                alignment = gan.module.generator.hidden_0[0].compute_alignment()
                print(alignment)
                alignment = gan.module.discriminator.hidden_2[0].compute_alignment()
                print(alignment)
                alignment = gan.module.discriminator.hidden_1[0].compute_alignment()
                print(alignment)
                alignment = gan.module.discriminator.hidden_0[0].compute_alignment()
                print(alignment)"""


        discriminator_loss_history.append(discriminator_batch_loss / (batch_idx + 1))
        generator_loss_history.append(generator_batch_loss / (batch_idx + 1))

        if plot:
            with torch.no_grad():

                fake_images = gan.module.generator(test_noise)

                size_figure_grid = 10
                fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
                for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
                    ax[i, j].get_xaxis().set_visible(False)
                    ax[i, j].get_yaxis().set_visible(False)

                for k in range(10 * 10):
                    i = k // 10
                    j = k % 10
                    ax[i, j].cla()
                    ax[i, j].imshow(
                        utils.im_convert(fake_images[k].view(1, dims, dims)),
                        # cmap='gray',
                        vmin=0,
                        vmax=1,
                    )

                label = 'Epoch {0}'.format(epoch)
                fig.text(0.5, 0.04, label, ha='center')
                plt.savefig(result_folder + "/gan%03d.png" % epoch)
                plt.show(block=False)
                plt.pause(1.5)
                plt.close(fig)

                # plot discriminator and generator loss history
                # clear figure
                plt.clf()
                if epoch == 1:
                    plt.scatter(0, discriminator_loss_history, label='discriminator loss')
                    plt.scatter(0, generator_loss_history, label='generator loss')
                else:
                    plt.plot(discriminator_loss_history, label='discriminator loss')
                    plt.plot(generator_loss_history, label='generator loss')
                plt.legend()
                plt.title("Loss at Epoch " + str(epoch))
                plt.xlim((-0.5, 10.5))
                plt.savefig(result_folder + "/loss-history"+str(epoch)+".png")
                plt.show(block=False)
                plt.pause(1.5)
                plt.close('all')

                """plt.hist(gan.module.generator.hidden_2[0].weight.detach().numpy().flatten(), alpha=0.5)
                plt.hist(gan.module.generator.hidden_2[0].weight_backward.detach().numpy().flatten(), alpha=0.5)
                plt.show()"""


    torch.save(gan.generator, "generator_model_sigmoid.pt" if USE_SIGMOID else "generator_model.pt")

    # plot discriminator and generator loss history
    # clear figure
    plt.clf()
    plt.plot(discriminator_loss_history, label='discriminator loss')
    plt.plot(generator_loss_history, label='generator loss')
    plt.savefig(result_folder + "/loss-history.png")
    plt.legend()
    plt.show()
    pass


class IdentityTransform(object):
    def __call__(self, tensor):
        return tensor


def basic_gan_experiment(data):
    transforms1 = transforms.Compose([
        transforms.ToTensor(),
        IdentityTransform() if USE_SIGMOID else transforms.Normalize((0.5,), (0.5,)),
    ])

    dataset: VisionDataset
    num_epochs: int
    dims: int
    dims_total: int

    if data == "MNIST":
        dataset = datasets.FashionMNIST(
            root="./data",
            train=True,
            download=True,
            transform=transforms1,
        )
        num_epochs = 10
        dims = 28
        dims_total = 28 * 28
        pass

    else:
        transforms2 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=1),
            IdentityTransform() if USE_SIGMOID else transforms.Normalize((0.5,), (0.5,)),
        ])
        dataset = datasets.CIFAR10(
            root="./data",
            train=True,
            download=True,
            transform=transforms2,
        )
        num_epochs = 30
        dims = 32
        dims_total = 32 * 32
        pass

    dataloader = DataLoader(dataset, batch_size=128,
                            drop_last=True,
                            shuffle=True)

    run_experiment(dataloader=dataloader, plot=True, num_epochs=num_epochs, dims=dims, dims_total=dims_total)
    pass


def advanced_gan_experiment():
    pass


if __name__ == '__main__':

    basic_gan_experiment(data="MNIST")

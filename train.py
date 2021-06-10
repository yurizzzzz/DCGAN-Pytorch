import torch
import itertools
import torch.nn as nn
from tqdm import tqdm
from model import dcgan
import torchvision.utils as utils
import torch.optim as optim
from torchvision import datasets
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

if __name__ == '__main__':
    input_fixed_noise = torch.randn((25, 100, 1, 1))
    input_fixed_noise = Variable(input_fixed_noise.cuda())

    transform = transforms.Compose([transforms.Resize(64), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    train_mnist = DataLoader(datasets.MNIST('data', train=True, download=True, transform=transform), batch_size=128,
                             shuffle=True)

    G = dcgan.generator()
    D = dcgan.discriminator()
    G.weight_init(mean=0.0, std=0.02)
    D.weight_init(mean=0.0, std=0.02)
    G.cuda()
    D.cuda()
    BCE_Loss = nn.BCELoss()

    G_optimizer = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    D_optimizer = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

    for epoch in range(1, 101):
        G_loss = []
        D_loss = []

        train_mnist = tqdm(train_mnist)
        for item, _ in train_mnist:
            mnist_batch = item.size()[0]

            y_real = torch.ones(mnist_batch)
            y_fake = torch.zeros(mnist_batch)

            item, y_real, y_fake = Variable(item.cuda()), Variable(y_real.cuda()), Variable(y_fake.cuda())
            D.zero_grad()
            D_result = D(item).squeeze()
            D_real_loss = BCE_Loss(D_result, y_real)

            rand_input = torch.randn((mnist_batch, 100, 1, 1))
            rand_input = Variable(rand_input.cuda())
            G_result = G(rand_input)

            D_result = D(G_result).squeeze()
            D_fake_loss = BCE_Loss(D_result, y_fake)

            D_train_loss = D_real_loss + D_fake_loss
            D_train_loss.backward()
            D_optimizer.step()
            D_loss.append(D_train_loss.item())

            G.zero_grad()
            rand_input = torch.randn((mnist_batch, 100, 1, 1))
            rand_input = Variable(rand_input.cuda())
            G_result = G(rand_input)
            D_result = D(G_result).squeeze()
            G_train_loss = BCE_Loss(D_result, y_real)
            G_train_loss.backward()
            G_optimizer.step()
            G_loss.append(G_train_loss.item())

            description = 'epoch: %d , D_loss: %.4f, G_loss: %.4f' % (
                epoch, torch.mean(torch.FloatTensor(D_loss)), torch.mean(torch.FloatTensor(G_loss)))
            train_mnist.set_description(description)
            train_mnist.update()

        G.eval()
        result = G(input_fixed_noise)
        G.train()
        utils.save_image(result[:24].detach().cpu(), '/home/result/' + str(epoch) + '.jpg')

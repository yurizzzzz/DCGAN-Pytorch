import torch
import codecs
import itertools
import torch.nn as nn
import torch.nn.init as init
from tqdm import tqdm
from model import face_dcgan
import torchvision.utils as utils
import torch.optim as optim
from torchvision import datasets
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


def init_ws_bs(m):
    if isinstance(m, nn.ConvTranspose2d):
        init.normal_(m.weight.data, std=0.2)
        init.normal_(m.bias.data, std=0.2)


if __name__ == '__main__':
    input_fixed_noise = torch.randn((32, 100, 1, 1))
    input_fixed_noise = Variable(input_fixed_noise.cuda())

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

    train_data = datasets.ImageFolder('./', transform)
    train_data = DataLoader(train_data, batch_size=100, shuffle=True)

    G = face_dcgan.generator()
    D = face_dcgan.discriminator()
    # G.weight_init(mean=0.0, std=0.02)
    # D.weight_init(mean=0.0, std=0.02)
    G.cuda()
    D.cuda()

    init_ws_bs(G)
    init_ws_bs(D)

    BCE_Loss = nn.BCELoss()

    G_optimizer = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    D_optimizer = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

    for epoch in range(1, 101):
        if epoch > 20:
            for par in G_optimizer.param_groups:
                par['lr'] = 0.00005
            for par in D_optimizer.param_groups:
                par['lr'] = 0.00005

        G_loss = []
        D_loss = []

        train_data = tqdm(train_data)
        for item, _ in train_data:
            train_batch = item.size()[0]

            y_real = torch.ones(train_batch)
            y_fake = torch.zeros(train_batch)

            item, y_real, y_fake = Variable(item.cuda()), Variable(y_real.cuda()), Variable(y_fake.cuda())
            D.zero_grad()
            D_result = D(item).squeeze()
            D_real_loss = BCE_Loss(D_result, y_real)

            rand_input = torch.randn((train_batch, 100, 1, 1))
            rand_input = Variable(rand_input.cuda())
            G_result = G(rand_input)

            D_result = D(G_result).squeeze()
            D_fake_loss = BCE_Loss(D_result, y_fake)

            D_train_loss = D_real_loss + D_fake_loss
            D_train_loss.backward()
            D_optimizer.step()
            D_loss.append(D_train_loss.item())

            G.zero_grad()
            rand_input = torch.randn((train_batch, 100, 1, 1))
            rand_input = Variable(rand_input.cuda())
            G_result = G(rand_input)
            D_result = D(G_result).squeeze()
            G_train_loss = BCE_Loss(D_result, y_real)
            G_train_loss.backward()
            G_optimizer.step()
            G_loss.append(G_train_loss.item())

            description = 'epoch: %d , D_loss: %.4f, G_loss: %.4f' % (
                epoch, torch.mean(torch.FloatTensor(D_loss)), torch.mean(torch.FloatTensor(G_loss)))
            train_data.set_description(description)
            train_data.update()

        with codecs.open('./g_loss.txt', mode='a', encoding='utf-8') as file_txt:
            g_loss = torch.mean(torch.FloatTensor(G_loss))
            g_loss = g_loss.numpy()[0]
            file_txt.write(str(g_loss) + '\n')
        with codecs.open('./d_loss.txt', mode='a', encoding='utf-8') as file_txt:
            d_loss = torch.mean(torch.FloatTensor(D_loss))
            d_loss = d_loss.numpy()[0]
            file_txt.write(str(d_loss) + '\n')

        G.eval()
        result = G(input_fixed_noise)
        G.train()
        utils.save_image(result[:32].detach().cpu(), '/home/result/' + str(epoch) + '.jpg')

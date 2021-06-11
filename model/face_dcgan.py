import torch.nn as nn


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(100, 1024, kernel_size=4, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(1024)
        self.deconv2 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(512)
        self.deconv3 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.deconv4 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.deconv5 = nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x1 = self.relu(self.bn1(self.deconv1(x)))
        x2 = self.relu(self.bn2(self.deconv2(x1)))
        x3 = self.relu(self.bn3(self.deconv3(x2)))
        x4 = self.relu(self.bn4(self.deconv4(x3)))
        output = self.tanh(self.deconv5(x4))

        return output

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

        # for m in self._modules:
        #     if isinstance(m, nn.ConvTranspose2d):
        #         nn.init.normal_(m.weight.data, mean, std)
        #
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.normal_(m.weight.data, mean, std)
        #         nn.init.constant_(m.bias.data, 0)


class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(512)
        self.conv4 = nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(1024)
        self.conv5 = nn.Conv2d(1024, 1, kernel_size=4, stride=1, padding=0)
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.leakyrelu(self.conv1(x))
        x2 = self.leakyrelu(self.bn1(self.conv2(x1)))
        x3 = self.leakyrelu(self.bn2(self.conv3(x2)))
        x4 = self.leakyrelu(self.bn3(self.conv4(x3)))
        output = self.sigmoid(self.conv5(x4))

        return output

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

        # for m in self._modules:
        #     if isinstance(m, nn.ConvTranspose2d):
        #         nn.init.normal_(m.weight.data, mean, std)
        #
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.normal_(m.weight.data, mean, std)
        #         nn.init.constant_(m.bias.data, 0)

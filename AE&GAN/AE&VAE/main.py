import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision import transforms, datasets
from matplotlib import pyplot as plt
from ae import AE
from vae import VAE
import numpy as np
import visdom


class myMSELoss(nn.Module):
    def __init__(self):
        super(myMSELoss, self).__init__()

    def forward(self, images1, images2):
        batch_size = images1.size(0)
        pixel_num = len(images1.view(-1)) / batch_size
        images1 = torch.mul(images1, 255)
        images2 = torch.pow(images2, 255)
        loss = 0
        for image_id in range(batch_size):
            image1 = images1[image_id]
            image2 = images2[image_id]
            image1 = image1.view(-1)
            image2 = image2.view(-1)
            dif = torch.sub(image1, image2)
            sqr = torch.pow(dif, 2)
            image_loss = torch.div(sqr, pixel_num)
            loss = torch.add(image_loss, loss)
        loss = torch.div(loss, batch_size)
        return torch.mean(loss)


def main():
    mnist_train = datasets.MNIST('../data', True, transform=transforms.Compose([
        transforms.ToTensor()
    ]), download=True)
    mnist_train = DataLoader(mnist_train, batch_size=32, shuffle=True)

    mnist_test = datasets.MNIST('../data', False, transform=transforms.Compose([
        transforms.ToTensor()
    ]), download=True)
    mnist_test = DataLoader(mnist_test, batch_size=32, shuffle=True)

    x, _ = iter(mnist_train).next()
    print('x:', x.shape)

    device = torch.device('cuda')
    model = AE().to(device)
    # model = VAE().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # print(model)

    viz = visdom.Visdom()
    viz.line([0], [-1], win='MSELoss', opts=dict(title='MSELoss'))
    globalStep = 0
    for epoch in range(1000):
        for batchidx, (x, _) in enumerate(mnist_train):
            # [b, 1, 28, 28]
            ############################################
            viz.images(x, nrow=8, win=f'x_origin{epoch}', opts=dict(title='x_origin'))

            ############################################
            x = x.to(device)

            x_hat, kld = model(x)
            ############################################
            viz.images(x_hat, nrow=8, win=f'x_reconstruction{epoch}', opts=dict(title='x_reconstruction'))
            ############################################
            loss = criterion(x_hat, x)

            if kld is not None:
                elbo = - loss - 1.0 * kld
                loss = - elbo
            globalStep += 1
            viz.line([loss.item()], [globalStep], win='MSELoss', update='append')
            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # print(epoch, 'loss:', loss.item(), 'kld:', kld.item())
        #
        # x, _ = iter(mnist_test).next()
        # x = x.to(device)
        # with torch.no_grad():
        #     x_hat, kld = model(x)
        # viz.images(x, nrow=8, win='x', opts=dict(title='x'))
        # viz.images(x_hat, nrow=8, win='x_hat', opts=dict(title='x_hat'))


if __name__ == '__main__':
    main()

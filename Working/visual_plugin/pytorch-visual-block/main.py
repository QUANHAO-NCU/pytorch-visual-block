import torch
from torch import nn, optim
import visdom

from lenet5 import Lenet5
from resnet import ResNet18
from dataset import myDataLoader


def main():
    batchsz = 2
    dataloader = myDataLoader()
    cifar_train = dataloader.get_data_set(mode='train', batch_size=batchsz)
    cifar_test = dataloader.get_data_set(mode='test', batch_size=batchsz)
    viz = visdom.Visdom()
    # viz.line([0], [-1], win='loss', opts=dict(title='loss'))
    # viz.line([0], [-1], win='test_acc', opts=dict(title='test_acc'))

    device = torch.device('cuda:0')
    # model = Lenet5().to(device)
    model = ResNet18().to(device)

    criteon = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # print(model)
    global_step = 0
    for epoch in range(1000):

        model.train()
        for batchidx, (x, label) in enumerate(cifar_train):
            x, label = x.to(device), label.to(device)
            logits = model(x)
            loss = criteon(logits, label)
            # viz.line([loss.item()], [global_step], win='loss', update='append')
            global_step += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(epoch, 'loss:', loss.item())

        model.eval()
        with torch.no_grad():
            total_correct = 0
            total_num = 0
            for x, label in cifar_test:
                x, label = x.to(device), label.to(device)
                logits = model(x)
                pred = logits.argmax(dim=1)
                correct = torch.eq(pred, label).float().sum().item()
                total_correct += correct
                total_num += x.size(0)

            acc = total_correct / total_num
            # viz.line([acc], [epoch], win='test_acc', update='append')
            print(epoch, 'test acc:', acc)


if __name__ == '__main__':
    main()

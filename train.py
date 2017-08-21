import argparse

from torch import optim
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from tqdm import tqdm
from tqdm import trange

from model import SqueezeNet


def train(epoch, data, squeeze_net, criterion, optimizer, args):
    train_set = DataLoader(data, batch_size=args.batch_size, num_workers=4, shuffle=True)

    progress_bar = tqdm(iter(train_set))
    moving_loss = 0

    squeeze_net.train(True)
    for x, y in progress_bar:
        x, y = Variable(x), Variable(y)

        if args.cuda:
            x, y = Variable(x).cuda(), Variable(y).cuda()

        output = squeeze_net(x, y)
        loss = criterion(output, y)
        squeeze_net.zero_grad()
        loss.backward()
        optimizer.step()

        if moving_loss == 0:
            moving_loss = loss.data[0]
        else:
            moving_loss = moving_loss * 0.9 + loss.data[0] * 0.1

        progress_bar.set_description(
            'Epoch: {}; Loss: {:.5f}; Avg: {:.5f}'.format(epoch + 1, loss.data[0], moving_loss))


def parse_args():
    parser = argparse.ArgumentParser(description='Train SqueezeNet with PyTorch.')
    parser.add_argument('--batch-size', action='store', type=int, dest='batch_size', default=256)
    parser.add_argument('--epochs', action='store', type=int, dest='epochs', default=90)
    parser.add_argument('--cuda', action='store', type=bool, dest='cuda', default=False)

    return parser.parse_args()


def main():
    args = parse_args()

    squeeze_net = SqueezeNet(10)
    if args.cuda:
        squeeze_net = squeeze_net.cuda()

    criterion = CrossEntropyLoss()
    optimizer = optim.Adam(squeeze_net.parameters(), lr=0.003)

    train_data = datasets.MNIST('./mnist_data', train=True, download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                                ]))

    valid_data = datasets.MNIST('./mnist_data', train=False, download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                                ]))

    for epoch in trange(args.epochs):
        train(epoch, train_data, squeeze_net, criterion, optimizer, args)


if __name__ == '__main__':
    main()

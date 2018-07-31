from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable


parser = argparse.ArgumentParser(description='VAE stl Example')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=2500, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.STL10('../data', split='train', download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.STL10('../data', split='test', transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)


# class VAE(nn.Module):
#     def __init__(self):
#         super(VAE, self).__init__()

#         self.fc1 = nn.Linear(96*96*3, 400*2)
#         self.fc21 = nn.Linear(400*2, 20*2)
#         self.fc22 = nn.Linear(400*2, 20*2)
#         self.fc3 = nn.Linear(20*2, 400*2)
#         self.fc4 = nn.Linear(400*2, 96*96*3)

#     def encode(self, x):
#         h1 = F.relu(self.fc1(x))
#         return self.fc21(h1), self.fc22(h1)

#     def reparameterize(self, mu, logvar):
#         if self.training:
#             std = torch.exp(0.5*logvar)
#             eps = torch.randn_like(std)
#             return eps.mul(std).add_(mu)
#         else:
#             return mu

#     def decode(self, z):
#         h3 = F.relu(self.fc3(z))
#         return F.sigmoid(self.fc4(h3))

#     def forward(self, x):
#         mu, logvar = self.encode(x.view(-1, 96*96*3))
#         z = self.reparameterize(mu, logvar)
#         return self.decode(z), mu, logvar


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # Encoder
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        # size 32x48x48

        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(32)
        # size 32x24x24

        self.conv5 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(32)
        self.conv6 = nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(16)
        # size 16x12x12

        self.fc1 = nn.Linear(12 * 12 * 16, 256)
        self.fc_bn1 = nn.BatchNorm1d(256)
        self.fc21 = nn.Linear(256, 64)
        self.fc22 = nn.Linear(256, 64)

        # Decoder
        self.fc3 = nn.Linear(64, 256)
        self.fc_bn3 = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, 12 * 12 * 16)
        self.fc_bn4 = nn.BatchNorm1d(12 * 12 * 16)
    
        self.inter1 = nn.ConvTranspose2d(16, 3, kernel_size=3, stride=1, padding=1, bias=False)
        # size 16x12x12

        self.conv7 = nn.ConvTranspose2d(16, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(32)
        self.conv8 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn8 = nn.BatchNorm2d(32)

        self.inter2 = nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1, bias=False) 
        # size 32x24x24

        self.conv9 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn9 = nn.BatchNorm2d(32)
        self.conv10 = nn.ConvTranspose2d(32, 32 , kernel_size=3, stride=1, padding=1, bias=False)    
        self.bn10 = nn.BatchNorm2d(32)

        self.inter3 = nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1, bias=False)
        # size 16x48x48

        self.conv11 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn11 = nn.BatchNorm2d(16)
        self.conv12 = nn.ConvTranspose2d(16, 3 , kernel_size=3, stride=1, padding=1, bias=False)
        # size 16x96x96

        self.relu = nn.ReLU()

    def encode(self, x):
        conv1 = self.relu(self.bn1(self.conv1(x)))
        conv2 = self.relu(self.bn2(self.conv2(conv1)))
        conv3 = self.relu(self.bn3(self.conv3(conv2)))
        conv4 = self.relu(self.bn4(self.conv4(conv3)))
        conv5 = self.relu(self.bn5(self.conv5(conv4)))
        conv6 = self.relu(self.bn6(self.conv6(conv5))).view(-1, 12 * 12 * 16)


        fc1 = self.relu(self.fc_bn1(self.fc1(conv6)))
        return self.fc21(fc1), self.fc22(fc1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        fc3 = self.relu(self.fc_bn3(self.fc3(z)))
        fc4 = self.relu(self.fc_bn4(self.fc4(fc3))).view(-1, 16, 12, 12)
        
        conv7 = self.relu(self.bn7(self.conv7(fc4)))
        conv8 = self.relu(self.bn8(self.conv8(conv7)))
        conv9 = self.relu(self.bn9(self.conv9(conv8)))
        conv10 = self.relu(self.bn10(self.conv10(conv9)))
        conv11 = self.relu(self.bn11(self.conv11(conv10)))

        final_level = F.sigmoid(self.conv12(conv11))#.view(-1, 3, 96, 96))
        level_3 = F.sigmoid(self.inter3(conv10))#.view(-1, 3, 96, 96))
        level_2 = F.sigmoid(self.inter2(conv8))#.view(-1, 3, 96, 96))
        level_1 = F.sigmoid(self.inter1(fc4))#.view(-1, 3, 96, 96))
        return level_1, level_2, level_3, final_level

    def forward(self, x):

        mu, logvar = self.encode(x)
        # print (mu.size(), logvar.size())
        z = self.reparameterize(mu, logvar)
        # print (z.size())
        # print (self.decode(z).size())
        l1, l2, l3, l4 = self.decode(z)
        # print (l1.size(), l2.size(), l3.size(), l4.size())
        return l1, l2, l3, l4, mu, logvar


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_l1, recon_l2, recon_l3, recon_l4, x, mu, logvar):
    
    BCE1 = F.binary_cross_entropy(recon_l1, F.adaptive_avg_pool2d(x, (12, 12)) , size_average=False)
    BCE2 = F.binary_cross_entropy(recon_l2, F.adaptive_avg_pool2d(x, (24, 24)), size_average=False)
    BCE3 = F.binary_cross_entropy(recon_l3, F.adaptive_avg_pool2d(x, (48, 48)), size_average=False)
    BCE4 = F.binary_cross_entropy(recon_l4, x, size_average=False)
    
    BCE = BCE1 + BCE2 + BCE3 + BCE4

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        l1, l2, l3, l4, mu, logvar = model(data)
        loss = loss_function(l1, l2, l3, l4, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            l1, l2, l3, l4, mu, logvar = model(data)
            test_loss += loss_function(l1, l2, l3, l4, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      l4.view(args.batch_size, 3, 96, 96)[:n]])
                save_image(comparison.cpu(),
                         './results_multires/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


for epoch in range(1, args.epochs + 1):
    train(epoch)
    test(epoch)
    with torch.no_grad():
        sample = torch.randn(args.batch_size, 64).to(device)
        l1, l2, l3, l4 = model.decode(sample)
        l4 = l4.cpu()
        save_image(l4.view(32, 3, 96, 96),
                   './results_multires/sample_' + str(epoch) + '.png')

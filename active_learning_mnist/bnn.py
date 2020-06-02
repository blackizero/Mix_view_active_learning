import torch
import torch.nn.functional as nnf
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
from torch.optim import SGD
from torch.distributions import constraints
import torchvision
import torchvision.transforms as torchvt
from torchvision.datasets.mnist import MNIST
from torch import nn
from pyro.infer import SVI, TraceMeanField_ELBO
import pyro
from pyro import poutine
import pyro.optim as pyroopt
import pyro.distributions as dist
import pyro.contrib.bnn as bnn
import matplotlib.pyplot as plt
import seaborn as sns
from torch.distributions.utils import lazy_property
import math


torch.set_default_tensor_type('torch.cuda.FloatTensor')

BATCH_SIZE = 128
LR = 0.002

train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,
    transform=torchvision.transforms.ToTensor(), # transfer to tensor
    download=False,
)
test_data = torchvision.datasets.MNIST(root='./mnist/', train=False, transform=torchvision.transforms.ToTensor(),)

train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE,shuffle=True)
test_loader = DataLoader(dataset=test_data,shuffle=True)


class BNN(nn.Module):
    def __init__(self, n_hidden=1024, n_classes=10):
        super(BNN, self).__init__()
        self.n_hidden = n_hidden
        self.n_classes = n_classes

    def model(self, images, labels=None, kl_factor=1.0):
        images = images.view(-1, 784)
        n_images = images.size(0)
        # Set-up parameters for the distribution of weights for each layer `a<n>`
        a1_mean = torch.zeros(784, self.n_hidden)
        a1_scale = torch.ones(784, self.n_hidden)
        a1_dropout = torch.tensor(0.25)
        a2_mean = torch.zeros(self.n_hidden + 1, self.n_classes)
        a2_scale = torch.ones(self.n_hidden + 1, self.n_hidden)
        a2_dropout = torch.tensor(1.0)
        a3_mean = torch.zeros(self.n_hidden + 1, self.n_classes)
        a3_scale = torch.ones(self.n_hidden + 1, self.n_hidden)
        a3_dropout = torch.tensor(1.0)
        a4_mean = torch.zeros(self.n_hidden + 1, self.n_classes)
        a4_scale = torch.ones(self.n_hidden + 1, self.n_classes)
        # Mark batched calculations to be conditionally independent given parameters using `plate`
        with pyro.plate('data', size=n_images):
            # Sample first hidden layer
            h1 = pyro.sample('h1', bnn.HiddenLayer(images, a1_mean, a1_dropout * a1_scale,
                                                   non_linearity=nnf.leaky_relu,
                                                   KL_factor=kl_factor))
            # Sample second hidden layer
            h2 = pyro.sample('h2', bnn.HiddenLayer(h1, a2_mean, a2_dropout * a2_scale,
                                                   non_linearity=nnf.leaky_relu,
                                                   KL_factor=kl_factor))
            # Sample third hidden layer
            h3 = pyro.sample('h3', bnn.HiddenLayer(h2, a3_mean, a3_dropout * a3_scale,
                                                   non_linearity=nnf.leaky_relu,
                                                   KL_factor=kl_factor))
            # Sample output logits
            logits = pyro.sample('logits', bnn.HiddenLayer(h3, a4_mean, a4_scale,
                                                           non_linearity=lambda x: nnf.log_softmax(x, dim=-1),
                                                           KL_factor=kl_factor,
                                                           include_hidden_bias=False))
            # One-hot encode labels
            labels = nnf.one_hot(labels) if labels is not None else None
            # Condition on observed labels, so it calculates the log-likehood loss when training using VI
            return pyro.sample('label', dist.OneHotCategorical(logits=logits), obs=labels)

    def guide(self, images, labels=None, kl_factor=1.0):
        images = images.view(-1, 784)
        n_images = images.size(0)
        # Set-up parameters to be optimized to approximate the true posterior
        # Mean parameters are randomly initialized to small values around 0, and scale parameters
        # are initialized to be 0.1 to be closer to the expected posterior value which we assume is stronger than
        # the prior scale of 1.
        # Scale parameters must be positive, so we constraint them to be larger than some epsilon value (0.01).
        # Variational dropout are initialized as in the prior model, and constrained to be between 0.1 and 1 (so dropout
        # rate is between 0.1 and 0.5) as suggested in the local reparametrization paper
        a1_mean = pyro.param('a1_mean', 0.01 * torch.randn(784, self.n_hidden))
        a1_scale = pyro.param('a1_scale', 0.1 * torch.ones(784, self.n_hidden),
                              constraint=constraints.greater_than(0.01))
        a1_dropout = pyro.param('a1_dropout', torch.tensor(0.25),
                                constraint=constraints.interval(0.1, 1.0))
        a2_mean = pyro.param('a2_mean', 0.01 * torch.randn(self.n_hidden + 1, self.n_hidden))
        a2_scale = pyro.param('a2_scale', 0.1 * torch.ones(self.n_hidden + 1, self.n_hidden),
                              constraint=constraints.greater_than(0.01))
        a2_dropout = pyro.param('a2_dropout', torch.tensor(1.0),
                                constraint=constraints.interval(0.1, 1.0))
        a3_mean = pyro.param('a3_mean', 0.01 * torch.randn(self.n_hidden + 1, self.n_hidden))
        a3_scale = pyro.param('a3_scale', 0.1 * torch.ones(self.n_hidden + 1, self.n_hidden),
                              constraint=constraints.greater_than(0.01))
        a3_dropout = pyro.param('a3_dropout', torch.tensor(1.0),
                                constraint=constraints.interval(0.1, 1.0))
        a4_mean = pyro.param('a4_mean', 0.01 * torch.randn(self.n_hidden + 1, self.n_classes))
        a4_scale = pyro.param('a4_scale', 0.1 * torch.ones(self.n_hidden + 1, self.n_classes),
                              constraint=constraints.greater_than(0.01))
        # Sample latent values using the variational parameters that are set-up above.
        # Notice how there is no conditioning on labels in the guide!
        with pyro.plate('data', size=n_images):
            h1 = pyro.sample('h1', bnn.HiddenLayer(images, a1_mean, a1_dropout * a1_scale,
                                                   non_linearity=nnf.leaky_relu,
                                                   KL_factor=kl_factor))
            h2 = pyro.sample('h2', bnn.HiddenLayer(h1, a2_mean, a2_dropout * a2_scale,
                                                   non_linearity=nnf.leaky_relu,
                                                   KL_factor=kl_factor))
            h3 = pyro.sample('h3', bnn.HiddenLayer(h2, a3_mean, a3_dropout * a3_scale,
                                                   non_linearity=nnf.leaky_relu,
                                                   KL_factor=kl_factor))
            logits = pyro.sample('logits', bnn.HiddenLayer(h3, a4_mean, a4_scale,
                                                           non_linearity=lambda x: nnf.log_softmax(x, dim=-1),
                                                           KL_factor=kl_factor,
                                                           include_hidden_bias=False))

    def infer_parameters(self, loader, lr=0.01, momentum=0.9,
                         num_epochs=30):
        optim = pyroopt.SGD({'lr': lr, 'momentum': momentum, 'nesterov': True})
        elbo = TraceMeanField_ELBO()
        svi = SVI(self.model, self.guide, optim, elbo)
        kl_factor = loader.batch_size / len(loader.dataset)
        for i in range(num_epochs):
            total_loss = 0.0
            total = 0.0
            correct = 0.0
            for images, labels in loader:
                loss = svi.step(images.cuda(), labels.cuda(), kl_factor=kl_factor)
                pred = self.forward(images.cuda(), n_samples=1).mean(0)
                total_loss += loss / len(loader.dataset)
                total += labels.size(0)
                correct += (pred.argmax(-1) == labels.cuda()).sum().item()
                param_store = pyro.get_param_store()
            print(f"[Epoch {i + 1}] loss: {total_loss:.5E} accuracy: {correct / total * 100:.5f}")

    def forward(self, images, n_samples=10):
        res = []
        for i in range(n_samples):
            t = poutine.trace(self.guide).get_trace(images)
            res.append(t.nodes['logits']['value'])
        return torch.stack(res, dim=0)


pyro.clear_param_store()
bayesnn = BNN()
bayesnn.infer_parameters(train_loader, num_epochs=10, lr=LR)

total = 0.0
correct = 0.0
for images, labels in test_loader:
    # print(labels.size())
    pred = bayesnn.forward(images.cuda().view(-1, 784), n_samples=1)
    total += labels.size(0)
    correct += (pred.argmax(-1) == labels.cuda()).sum().item()
print(f"Test accuracy: {correct / total * 100:.5f}")

uncertain_images = []
for image, _ in test_loader:
    n_samples = 30
    preds = bayesnn.forward(image.cuda().view(-1, 784), n_samples=n_samples).argmax(-1).argmax(-1).squeeze()
    print(preds.size())
    pred_sum = [(i, c) for i, c in enumerate(preds.bincount(minlength=10).tolist()) if c > 0]
    if len(pred_sum) > 1:
        uncertain_images.append((image, "\n".join(f"{i}: {c / (n_samples):.2f}" for i, c in pred_sum)))
    if len(uncertain_images) >= 64:
        break

fig, axes = plt.subplots(nrows=8, ncols=8)
fig.subplots_adjust(hspace=2.0)
for i, (image, label) in enumerate(uncertain_images):
    axes[i // 8][i % 8].imshow(image[0][0], cmap='gray')
    axes[i // 8][i % 8].set_title(f"{label}")
    axes[i // 8][i % 8].set_xticks([])
    axes[i // 8][i % 8].set_yticks([])
fig.set_size_inches(8, 16)
fig.tight_layout()
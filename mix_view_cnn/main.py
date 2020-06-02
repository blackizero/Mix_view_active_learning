import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.nn.functional as F
#from torchsummary import summary

import numpy as np
import time
import os
import argparse
from scipy.stats import mode

from models.MVCNN import *
from models.RESNET import *
from utils.util import logEpoch, save_checkpoint
from utils.custom_dataset import MultiViewDataset
from utils.logger import Logger

MVCNN = 'mvcnn'
RESNET = 'resnet'
MODELS = [MVCNN, RESNET]

parser = argparse.ArgumentParser(description='MVCNN-PyTorch')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--model', '-m', metavar='MODEL', default=MVCNN, choices=MODELS,
                    help='pretrained model: ' + ' | '.join(MODELS) + ' (default: {})'.format(MVCNN))
parser.add_argument('--depth', choices=[18, 34, 50, 101, 152], type=int, metavar='N', default=18, help='resnet depth (default: resnet18)')
parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run (default: 100)')
parser.add_argument('-b', '--batch-size', default=4, type=int,
                    metavar='N', help='mini-batch size (default: 4)')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate (default: 0.0001)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum (default: 0.9)')
parser.add_argument('--lr-decay-freq', default=30, type=float,
                    metavar='W', help='learning rate decay (default: 30)')
parser.add_argument('--lr-decay', default=0.1, type=float,
                    metavar='W', help='learning rate decay (default: 0.1)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-r', '--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')

args = parser.parse_args()

print('Loading Multi-view data')

transform = transforms.Compose([
    transforms.CenterCrop(500),
    transforms.Resize(224),
    transforms.ToTensor(),
])

device = torch.device("cuda:2,3" if torch.cuda.is_available() else "cpu")

# Load dataset
dset_train = MultiViewDataset(args.data, 'train', transform=transform)
train_loader = DataLoader(dset_train, batch_size=args.batch_size, shuffle=True, num_workers=2)

dset_val = MultiViewDataset(args.data, 'test', transform=transform)

val_loader = DataLoader(dset_val, batch_size=args.batch_size, shuffle=True, num_workers=2)

classes = dset_train.classes
print(len(classes), classes)

if args.model == RESNET:
    if args.depth == 18:
        model = resnet18(pretrained=args.pretrained, num_classes=len(classes))
    elif args.depth == 34:
        model = resnet34(pretrained=args.pretrained, num_classes=len(classes))
    elif args.depth == 50:
        model = resnet50(pretrained=args.pretrained, num_classes=len(classes))
    elif args.depth == 101:
        model = resnet101(pretrained=args.pretrained, num_classes=len(classes))
    elif args.depth == 152:
        model = resnet152(pretrained=args.pretrained, num_classes=len(classes))
    else:
        raise Exception('Specify number of layers for resnet in command line. --resnet N')
    print('Using ' + args.model + str(args.depth))
else:
    model = mvcnn(pretrained=args.pretrained,num_classes=len(classes))
    print('Using ' + args.model)

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

model.to(device)
#print(summary(model, (12, 3, 224, 224)))
cudnn.benchmark = True

logger = Logger('logs')

# Loss and Optimizer
lr = args.lr
n_epochs = args.epochs
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

best_acc = 0.0
best_loss = 0.0
start_epoch = 0


# Helper functions
def load_checkpoint():
    global best_acc, start_epoch
    # Load checkpoint.
    print('\n==> Loading checkpoint..')
    assert os.path.isfile(args.resume), 'Error: no checkpoint file found!'

    checkpoint = torch.load(args.resume)
    best_acc = checkpoint['best_acc']
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


def train():
    train_size = len(train_loader)
    for i, (inputs, targets) in enumerate(train_loader):
        # Convert from list of 3D to 4D
        inputs = np.stack(inputs, axis=1)

        inputs = torch.from_numpy(inputs)

        inputs, targets = inputs.cuda(device), targets.cuda(device)
        inputs, targets = Variable(inputs), Variable(targets)

        # compute output
        m_outputs, outputs_pool = model(inputs)
        m_loss = criterion(m_outputs, targets)
        s_loss = 0.0
        for v in outputs_pool:
            s_loss += criterion(v, targets)
        loss = m_loss + s_loss
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % args.print_freq == 0:
            print("\tIter [%d/%d] Loss: %.4f M_loss: %.4f S_loss: %.4f" % (i + 1, train_size, loss.item(), m_loss.item(), s_loss.item()))


# Validation and Testing
def eval(data_loader, is_test=False):
    if is_test:
        load_checkpoint()

    # Eval
    total = 0.0
    correct = 0.0
    correct_single = 0.0

    total_loss = 0.0
    n = 0
    v = 0
    num_samples = 50
    conf = []
    Variation = np.zeros(shape=len(data_loader.dataset))

    for i, (inputs, targets) in enumerate(data_loader):
        with torch.no_grad():
            # Convert from list of 3D to 4D
            inputs = np.stack(inputs, axis=1)

            inputs = torch.from_numpy(inputs)

            inputs, targets = inputs.cuda(device), targets.cuda(device)
            inputs, targets = Variable(inputs), Variable(targets)

            # compute output
            m_outputs, outputs_pool = model(inputs)
            m_loss = criterion(m_outputs, targets)
            s_loss = 0.0
            for v in outputs_pool:
                s_loss += criterion(v, targets)
            loss = m_loss + s_loss

            total_loss += loss
            n += 1

            # multi-view results
            _, predicted = torch.max(m_outputs.data, 1)
            total += targets.size(0)
            correct += (predicted.cpu() == targets.cpu()).sum()
            # single-view results
            for v in outputs_pool:
                _, predicted_s = torch.max(v.data, 1)
                correct_single += (predicted_s.cpu() == targets.cpu()).sum()

    avg_test_acc = 100 * correct / total
    avg_loss = total_loss / n
    avg_test_acc_single = 100 * correct_single / (total * len(outputs_pool))
    return avg_test_acc, avg_test_acc_single, avg_loss

""""
    for i, (inputs, targets) in enumerate(data_loader):
        with torch.no_grad():
            # Convert from list of 3D to 4D
            inputs = np.stack(inputs, axis=1)

            inputs = torch.from_numpy(inputs)

            inputs, targets = inputs.cuda(device), targets.cuda(device)
            inputs, targets = Variable(inputs), Variable(targets)

            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss


            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            #correct += (predicted.cpu() == targets.cpu()).sum()

            yhats = [model(inputs).data for _ in range(num_samples)]
            preds = F.softmax(torch.stack(yhats), dim=1)

            results = torch.topk(preds.cpu().data, k=1, dim=1)
            conf.append(results[0][0].numpy())
            preds = torch.stack(yhats).argmax(-1).cpu().data.numpy()
            preds, Mode = mode(preds)
            preds = np.squeeze(preds, axis=0)
            print(preds.shape)
            preds = torch.from_numpy(preds)
            print('preds value', preds)
            print('target value', targets.cpu())
            Mode = Mode / num_samples
            for t in range(targets.size(0)):
                Variation[v] = Mode[0][t]
                v = v + 1
            print('mode confidence vaule', Variation)
            print((preds.cpu() == targets.cpu()).sum())
            correct += (preds.cpu() == targets.cpu()).sum()

            n += 1


    avg_test_acc = 100 * correct / total
    avg_loss = total_loss / n

    p_hat = np.array(conf)
    confidence_mean = np.mean(p_hat, axis=0)
    confidence_var = np.var(p_hat, axis=0)
    epistemic = np.mean(p_hat ** 2, axis=0) - np.mean(p_hat, axis=0) ** 2
    aleatoric = np.mean(p_hat * (1 - p_hat), axis=0)

    print("Mean is: ", np.mean(confidence_mean))
    print("Variance is: ", np.mean(confidence_var))
    print("Epistemic Uncertainity is: ", np.mean(epistemic))
    print("Aleatoric Uncertainity is: ", np.mean(aleatoric))

"""


# Training / Eval loop
if args.resume:
    load_checkpoint()

for epoch in range(start_epoch, n_epochs):
    print('\n-----------------------------------')
    print('Epoch: [%d/%d]' % (epoch+1, n_epochs))
    start = time.time()

    model.train()
    train()
    print('Time taken: %.2f sec.' % (time.time() - start))

    # delet model.eval() using model.train()
    model.eval()
    avg_test_acc, avg_test_acc_single, avg_loss = eval(val_loader)

    print('\nEvaluation:')
    print('\tVal Acc of Multi-view: %.2f - Val Acc of Single view: %.2f - Loss: %.4f' % (avg_test_acc.item(), avg_test_acc_single.item(), avg_loss.item()))
    print('\tCurrent best val acc: %.2f' % best_acc)

    # Log epoch to tensorboard
    # See log using: tensorboard --logdir='logs' --port=6006
    logEpoch(logger, model, epoch + 1, avg_loss, avg_test_acc)

    # Save model
    if avg_test_acc > best_acc:
        print('\tSaving checkpoint - Acc: %.2f' % avg_test_acc)
        best_acc = avg_test_acc
        best_loss = avg_loss
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'acc': avg_test_acc,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, args.model)

    # Decaying Learning Rate
    if (epoch + 1) % args.lr_decay_freq == 0:
        lr *= args.lr_decay
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        print('Learning rate:', lr)

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
#from torchsummary import summary

import numpy as np
import time
import os
import random
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
parser.add_argument('--acqu', '-a', metavar='acqu', default='MAX_ENTROPY', type=str,
                    help='acquisition functions UNCERTAINTY BALD VAR_RATIOS MAX_ENTROPY RANDOM)')
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

#os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load dataset
dset_train = MultiViewDataset(args.data, 'train', transform=transform)
shuffler_idx = list(range(len(dset_train)))
random.shuffle(shuffler_idx)
labeled_set = shuffler_idx[:200]
unlabeled_set = shuffler_idx[200:]

train_loader = DataLoader(dset_train, batch_size=args.batch_size, sampler=SubsetRandomSampler(labeled_set),
                                  pin_memory=True, num_workers=0)

dset_unlabeled = MultiViewDataset(args.data, 'train', transform=transform)


dset_val = MultiViewDataset(args.data, 'test', transform=transform)
val_loader = DataLoader(dset_val, batch_size=args.batch_size, shuffle=True, num_workers=0)

classes = dset_train.classes
print(len(dset_train))
print(len(dset_val))
print(len(classes), classes)

def init_model():
    global model
    global criterion
    global optimizer
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
        model = mvcnn(pretrained=args.pretrained, num_classes=len(classes))
        print('Using ' + args.model)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model.to(device)
    # print(summary(model, (n_views, 3, 224, 224)))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

cudnn.benchmark = True

# logger = Logger('logs')

# Loss and Optimizer
lr = args.lr
n_epochs = args.epochs


best_acc = 0.0
best_loss = 0.0
start_epoch = 0
pool_size = 2000
n_dropout = 20
n_views = 12



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
    total_m_loss = 0.0
    total_s_loss = 0.0
    n = 0

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
            total_m_loss += m_loss
            total_s_loss += s_loss
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
    avg_m_loss = total_m_loss / n
    avg_s_loss = total_s_loss / n
    avg_test_acc_single = 100 * correct_single / (total * len(outputs_pool))
    return avg_test_acc, avg_test_acc_single, avg_loss, avg_m_loss, avg_s_loss

def val_acqu(data_loader, predict_classes=False):
    # Eval using MC dropout
    model.train()
    predictions = []
    if predict_classes:
        predictions_s = np.zeros(shape=(n_views, args.batch_size))
    else:
        predictions_s = np.zeros(shape=(n_views, args.batch_size, len(classes)))
    for i, (inputs, targets) in enumerate(data_loader):
        with torch.no_grad():
            # Convert from list of 3D to 4D
            inputs = np.stack(inputs, axis=1)
            inputs = torch.from_numpy(inputs)
            inputs, targets = inputs.cuda(device), targets.cuda(device)
            inputs, targets = Variable(inputs), Variable(targets)

            # compute output
            yhats, yhats_s = model(inputs)
            preds = F.softmax(yhats.cpu())
            views = []
            for v in yhats_s:
                views.append(F.softmax(v.cpu()).data.numpy())
            views = np.array(views)
            if predict_classes:
                predictions.extend(np.argmax(preds.data.numpy(), axis=-1))
                predictions_s = np.concatenate((predictions_s, np.argmax(views, axis=-1)), axis=1)
            else:
                predictions.extend(preds.data.numpy())
                predictions_s = np.concatenate((predictions_s, views), axis=1)
    if predict_classes:
        predictions_s = predictions_s[:, args.batch_size:]
    else:
        predictions_s = predictions_s[:,args.batch_size:,:]
    return predictions, predictions_s

def getAcquisitionFunction(name):
    if name == "MAX_ENTROPY":
        return max_entroy_acquisition # entropy from mvcnn
    elif name == "MAX_MIX_ENTROPY":
        return max_mix_entroy_acquisition # entropy form mv+sv cnn
    elif name == "VAR_RATIOS":
        return variation_ratios_acquisition # entropy from mvcnn
    elif name == "VAR_MIX_RATIOS":
        return variation_mix_ratios_acquisition # entropy from mv+sv cnn
    else:
        print ("ACQUSITION FUNCTION NOT IMPLEMENTED")

def variation_ratios_acquisition(model, unlabeled_loader):
    print("VARIATIONAL RATIOS ACQUSITION FUNCTION")
    # using model.train() instead of model.eval(), needed for MC dropout
    model.train()
    dropout_iterations = n_dropout
    All_Dropout_Classes = np.zeros(shape=(pool_size, 1))
    start_time = time.time()
    for d in range(dropout_iterations):
        predictions,_ = val_acqu(unlabeled_loader, predict_classes=True)
        predictions = np.array(predictions)
        predictions = np.expand_dims(predictions, axis=1)
        All_Dropout_Classes = np.append(All_Dropout_Classes, predictions, axis=1)
    print("Dropout Iterations took --- %s seconds ---" % (time.time() - start_time))
    # print (All_Dropout_Classes)
    Variation = np.zeros(shape=(pool_size))
    for t in range(pool_size):
        L = np.array([0])
        for d_iter in range(dropout_iterations):
            L = np.append(L, All_Dropout_Classes[t, d_iter + 1])
        Predicted_Class, Mode = mode(L[1:])
        v = np.array([1 - Mode / float(dropout_iterations)])
        Variation[t] = v
    points_of_interest = Variation.flatten()
    print(points_of_interest)
    return points_of_interest

def variation_mix_ratios_acquisition(model, unlabeled_loader):
    print("VARIATIONAL MIX RATIOS ACQUSITION FUNCTION")
    # using model.train() instead of model.eval(), needed for MC dropout
    model.train()
    dropout_iterations = n_dropout
    All_Dropout_Classes = np.zeros(shape=(pool_size, 1))
    All_Dropout_Classes_s = np.zeros(shape=(n_views, pool_size, 1))
    start_time = time.time()
    for d in range(dropout_iterations):
        predictions, predictions_s = val_acqu(unlabeled_loader, predict_classes=True)
        predictions = np.array(predictions)
        predictions = np.expand_dims(predictions, axis=1)
        predictions_s = np.expand_dims(predictions_s, axis=2)
        All_Dropout_Classes = np.append(All_Dropout_Classes, predictions, axis=1)
        All_Dropout_Classes_s = np.append(All_Dropout_Classes_s, predictions_s, axis=2)
    print("Dropout Iterations took --- %s seconds ---" % (time.time() - start_time))
    # print (All_Dropout_Classes)
    All_Dropout_Classes_s = All_Dropout_Classes_s[:,:,1:]
    All_Dropout_Classes_s = np.reshape(All_Dropout_Classes_s,(pool_size,-1))
    Variation = np.zeros(shape=(pool_size))
    for t in range(pool_size):
        L = np.array([0])
        for d_iter in range(dropout_iterations):
            L = np.append(L, All_Dropout_Classes[t, d_iter + 1])
        for d_iter in range(dropout_iterations*n_views):
            L_s = np.append(L, All_Dropout_Classes_s[t, d_iter])
        Predicted_Class, Mode = mode(L[1:])
        v = np.array([1 - sum(L_s==Predicted_Class) / float(dropout_iterations*n_views)])
        Variation[t] = v
    points_of_interest = Variation.flatten()
    print(points_of_interest)
    return points_of_interest

def max_entroy_acquisition(model, unlabeled_loader):
    print("MAX ENTROPY ACQUSITION FUNCTION")
    # using model.train() instead of model.eval(), need for MC dropout
    model.train()

    dropout_iterations = n_dropout
    score_All = np.zeros(shape=(pool_size, len(classes)))
    start_time = time.time()
    for d in range(dropout_iterations):
        predictions,_ = val_acqu(unlabeled_loader)
        predictions = np.array(predictions)
        score_All = score_All + predictions
    print("Dropout Iterations took --- %s seconds ---" % (time.time() - start_time))
    # print (All_Dropout_Classes)
    Avg_Pi = np.divide(score_All, dropout_iterations)
    Log_Avg_Pi = np.log2(Avg_Pi)
    Entropy_Avg_Pi = - np.multiply(Avg_Pi, Log_Avg_Pi)
    Entropy_Average_Pi = np.sum(Entropy_Avg_Pi, axis=1)
    U_X = Entropy_Average_Pi
    points_of_interest = U_X.flatten()
    print(points_of_interest.shape)

    return points_of_interest

def max_mix_entroy_acquisition(model, unlabeled_loader):
    print("MAX ENTROPY MIX ACQUSITION FUNCTION")
    # using model.train() instead of model.eval(), need for MC dropout
    model.train()

    dropout_iterations = n_dropout
    score_All = np.zeros(shape=(pool_size, len(classes)))
    score_All_view = np.zeros(shape=(n_views, pool_size, len(classes)))
    start_time = time.time()
    for d in range(dropout_iterations):
        predictions, predictions_s = val_acqu(unlabeled_loader)
        predictions = np.array(predictions)
        predictions_s = np.array(predictions_s)
        score_All_view = score_All_view + predictions_s
        score_All = score_All + predictions
    print("Dropout Iterations took --- %s seconds ---" % (time.time() - start_time))
    # print (All_Dropout_Classes)
    Avg_Pi = np.divide(score_All, dropout_iterations)
    Log_Avg_Pi = np.log2(Avg_Pi)
    Entropy_Avg_Pi = - np.multiply(Avg_Pi, Log_Avg_Pi)
    Entropy_Average_Pi = np.sum(Entropy_Avg_Pi, axis=1)
    U_X = Entropy_Average_Pi
    print('multi entropy')
    print(U_X)
    U_X_S = np.zeros(shape=(U_X.shape))
    for view in score_All_view:
        Avg_Pi = np.divide(view, dropout_iterations)
        Log_Avg_Pi = np.log2(Avg_Pi)
        Entropy_Avg_Pi = - np.multiply(Avg_Pi, Log_Avg_Pi)
        Entropy_Average_Pi = np.sum(Entropy_Avg_Pi, axis=1)
        U_X_S += Entropy_Average_Pi
    Entropy_Average_Pi = np.divide(U_X_S, n_views)
    print('single entropy')
    print(Entropy_Average_Pi)
    U_total = U_X + Entropy_Average_Pi
    points_of_interest = U_total.flatten()
    print(points_of_interest)

    return points_of_interest

results_acc_mv = []
results_acc_sv = []
results_loss_mv = []
results_loss_sv = []
N_cycle = 40
# Active learning
for cycle in range(N_cycle):
    init_model()
    print('Initialize model----------')
    # Training / Eval loop
    #if args.resume:
    #    load_checkpoint()

    for epoch in range(start_epoch, n_epochs):
        print('\n-----------------------------------')
        print('Epoch: [%d/%d]' % (epoch + 1, n_epochs))
        start = time.time()
        model.train()
        train()
        print('Time taken: %.2f sec.' % (time.time() - start))
        # Decaying Learning Rate
        if (epoch + 1) % args.lr_decay_freq == 0:
            lr *= args.lr_decay
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            print('Learning rate:', lr)


    # model eval
    model.eval()
    avg_test_acc, avg_test_acc_single, avg_loss, avg_m_loss, avg_s_loss = eval(val_loader)
    print('\nEvaluation:')
    print('Cycle {}/{} || Label set size {}'.format(cycle + 1, N_cycle, len(labeled_set)))
    print(
        '\tVal Acc of Multi-view: %.2f - Val Acc of Single view: %.2f - Loss: %.4f - M_loss: %.4f - S_loss: %.4f' % (
            avg_test_acc.item(), avg_test_acc_single.item(),
            avg_loss.item(), avg_m_loss.item(), avg_s_loss.item()))
    results_acc_mv.append(avg_test_acc.item())
    results_acc_sv.append(avg_test_acc_single.item())
    print('acc_mv', results_acc_mv)
    print('acc_sv', results_acc_sv)
    results_loss_mv.append(avg_m_loss.item())
    results_loss_sv.append(avg_s_loss.item())
    print('loss_mv', results_loss_mv)
    print('loss_sv', results_loss_sv)


    #  Update the labeled dataset via uncertainty estiamtion
    # Randomly sample pool_size unlabeled data points
    random.shuffle(unlabeled_set)
    subset = unlabeled_set[:pool_size]

    # Create unlabeled dataloader for the unlabeled subset
    unlabeled_loader = DataLoader(dset_train, batch_size=args.batch_size, sampler=SubsetRandomSampler(subset), pin_memory=True, num_workers=0)

    # random selection
    if args.acqu == "RANDOM":
        arg = np.array(range(0,pool_size))
    else:
        acquisition_function = getAcquisitionFunction(args.acqu)
        # Measure uncertainty of each data points in the subset
        uncertainty = acquisition_function(model, unlabeled_loader)
        # Index in ascending order
        print(uncertainty[:200])
        arg = np.argsort(uncertainty)
        print(arg)
        print('11111')
        print(arg[:200])
        print('22222')
        print(arg[-200:])

    # Update the labeled dataset and the unlabeled dataset, respectively
    labeled_set += list(torch.tensor(subset)[arg][-200:].numpy())
    unlabeled_set = list(torch.tensor(subset)[arg][:-200].numpy()) + unlabeled_set[pool_size:]

    # Create a new dataloader for the updated labeled dataset
    train_loader = DataLoader(dset_train, batch_size=args.batch_size, sampler=SubsetRandomSampler(labeled_set), pin_memory=True, num_workers=0)


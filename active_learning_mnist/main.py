import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import torch.utils.data as Data
import numpy as np
import math
import time
import random
import scipy.ndimage
import argparse
from scipy.stats import mode

parser = argparse.ArgumentParser(description='ACTIVE_LEARNING')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--acqu', '-a', metavar='acqu', default='RANDOM', type=str,
                    help='acquisition functions(RANDOM(default) BALD VAR_RATIOS MAX_ENTROPY MEAN_STD)')
parser.add_argument('--augment', type=int, default=1, metavar='DA',
                    help='augments labeled data (default: True)')
parser.add_argument('--cuda', type=bool, default=False,
                    help='enables CUDA training (default: True)')
parser.add_argument('--supervised', type=int, default=0,
                    help='enables use of unlabeled data for semi-supervised learning (default True)')
parser.add_argument("--tsa",default='linear',type=str,
                    help="Set the method to perform threshold annealing on supervised data")

args = parser.parse_args()
print(args)


# hyper-params
epochs = 100             #number of epoch per training session
LR = 0.001
nb_classes = 10
tsa = False
verbose = False


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def to_gpu(var, cuda):
    if cuda:
        return var.cuda(device)
    return var


def kl_for_log_probs(log_p, log_q):
    p = torch.exp(log_p)
    neg_ent = torch.sum(p * log_p, dim=-1)
    neg_cross_ent = torch.sum(p * log_q, dim=-1)
    kl = neg_ent - neg_cross_ent
    return kl


def get_tsa_threshold(global_step, num_train_step, start, end, schedule='linear', scale=5):
    '''
    Schedule: Must be either linear, log or exp. Defines the type of schedule used for the annealing.
    start = 1 / K , K being the number of classes
    end = 1
    scale = exp(-scale) close to zero
    '''
    assert schedule in ['linear', 'log', 'exp']
    training_progress = global_step / num_train_step

    if schedule == 'linear':
        threshold = training_progress
    elif schedule == 'exp':
        threshold = torch.exp((training_progress - 1) * scale)
    elif schedule == 'log':
        threshold = 1 - torch.exp((-training_progress) * scale)
    return threshold * (end - start) + start


# Crops the center of the image
def crop_center(img,cropx,cropy):
    if img.ndim == 2:
        y,x = img.shape
        startx = x//2-(cropx//2)
        starty = y//2-(cropy//2)
        return img[starty:starty+cropy,startx:startx+cropx]
    elif img.ndim == 3:
        b,y,x = img.shape
        startx = x//2-(cropx//2)
        starty = y//2-(cropy//2)
        return img[:,starty:starty+cropy,startx:startx+cropx]

# Take a random crop of the image
def crop_random(img,cropx,cropy):
    # takes numpy input
    if img.ndim == 2:
        x1 = random.randint(0, img.shape[0] - cropx)
        y1 = random.randint(0, img.shape[1] - cropy)
        return img[x1:x1+cropx,y1:y1+cropy]

# Image data augmentation
def augment(image):
    npImg = image.numpy().squeeze(1)
    # rotate image by maximum of 25 degrees clock- or counter-clockwise
    rotation = [random.randrange(-25,25) for i in range(len(image))]
    rotatedImg = [scipy.ndimage.interpolation.rotate(im, rotation[i], axes=(0,1)) for i, im in enumerate(npImg)]
    # crop image to 28x28 as rotation increases size
    rotatedImgCentered = [crop_center(im, 28, 28) for im in rotatedImg]
    # pad image by 3 pixels on each edge (-0.42421296 background color)
    paddedImg = [np.pad(im, 3, 'constant',constant_values=-0.42421296) for im in rotatedImgCentered]
    # randomly crop from padded image
    cropped = np.array([crop_random(im, 28, 28) for im in paddedImg])
    return torch.FloatTensor(cropped).unsqueeze(1)


train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,
    transform=torchvision.transforms.ToTensor(), # transfer to tensor
    download=True,
)
test_data = torchvision.datasets.MNIST(root='./mnist/', train=False, transform=torchvision.transforms.ToTensor(),)

train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size,shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size,shuffle=True)

train_data_all = train_loader.dataset.train_data
train_target_all = train_loader.dataset.train_labels
shuffler_idx = torch.randperm(train_target_all.size(0))
train_data_all = train_data_all[shuffler_idx]
train_target_all = train_target_all[shuffler_idx]

train_data = []
train_target = []
train_data_val = train_data_all[:100, :, :]
train_target_val = train_target_all[:100]
train_data_pool = train_data_all[1000:, :, :]
train_target_pool = train_target_all[1000:]

train_data_val=train_data_val.unsqueeze(1)
train_data_pool=train_data_pool.unsqueeze(1)
train_data_all=train_data_all.unsqueeze(1)

train_data_pool = train_data_pool.float()
train_data_val = train_data_val.float()
train_data_all = train_data_all.float()
for i in range(0,10):
    arr = np.array(np.where(train_target_all.numpy()==i))
    idx = np.random.permutation(arr)
    data_i = train_data_all.numpy()[idx[0][0:2],:,:]   # pick first 2 elements of shuffled idx array
    target_i = train_target_all.numpy()[idx[0][0:2]]
    train_data.append(data_i)
    train_target.append(target_i)

train_data = np.concatenate(train_data, axis=0).astype("float32")
train_target = np.concatenate(train_target, axis=0)
train_data = torch.from_numpy(train_data/255)
train_target = torch.from_numpy(train_target)
val_data = train_data_val/255
val_target = train_target_val
pool_data = train_data_pool/255
pool_target = train_target_pool


# initialize dataset

def initialize_train_set():
    # Training Data set
    global train_loader
    global train_data
    train = Data.TensorDataset(train_data, train_target)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)

def initialize_train_pool_set():
    # Training Data set
    global train_pool_loader
    global pool_data

    train_pool = Data.TensorDataset(pool_data, pool_target)
    train_pool_loader = DataLoader(train_pool, batch_size=args.batch_size, shuffle=True)

def initialize_val_set():
    global val_loader
    global val_data
    #Validation Dataset

    val = Data.TensorDataset(val_data, val_target)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=True)

initialize_train_set()
initialize_train_pool_set()
initialize_val_set()


class Net(nn.Module):
    def __init__(self, input_shape=(1, 28, 28)):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4),
            nn.Dropout2d(0.25),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=4),
            nn.Dropout2d(0.25),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25), )

        input_size = self._get_size(input_shape)
        self.dense = nn.Sequential(nn.Linear(input_size, 256))
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, nb_classes))

    def _get_size(self, shape):
        input = Variable(torch.rand(args.batch_size, *shape))
        output_feat = self.conv(input)
        n_size = output_feat.data.view(args.batch_size, -1).size(1)
        return n_size

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(self.dense(x))
        return x


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # input shape(1,28,28)
            nn.Conv2d(
                in_channels=1,  # input height
                out_channels=16,  # n_filters
                kernel_size=5,  # filter size
                stride=1,  # filter movement/step
                padding=2,
            ),  # output shape(16,28,28)
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=2),  # output shape (16,14,14)
        )

        self.conv2 = nn.Sequential(  # input shape(16,14,14)
            nn.Conv2d(16, 32, 5, 1, 2),  # output (32,14,14)
            nn.ReLU(),
            nn.MaxPool2d(2),  # output (32,7,7)
        )

        self.out = nn.Linear(32 * 7 * 7, 10)  # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # unfold conv map to (batch_size, 32*7*7)
        output = self.out(x)
        return output



def train(epoch):
    model.train()
    loss = None
    global train_data
    # update global step per epoch
    global_step = (epoch-1)*(len(train_data) / args.batch_size)
    num_train_step = 50 * len(train_data) / args.batch_size
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.augment:
            data = augment(data)  # augment labeled data
            # print("using augumentation")
        if args.cuda:
            data, target = data.cuda(device), target.cuda(device)
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        softmaxed = F.softmax(output, dim=-1)
        criterion = nn.CrossEntropyLoss(reduction = 'none')

        loss = criterion(output, target)

        if tsa:
            tsa_start = 1./nb_classes
            tsa_threshold = get_tsa_threshold(global_step=global_step, \
                                              num_train_step=num_train_step, start=tsa_start, \
                                              end=1., schedule='linear', scale=5)
            loss_mask = torch.ones(loss.size()).long()
            probas = torch.gather(softmaxed, dim=-1, index=target.unsqueeze(1)).cpu()
            loss_mask = torch.where(probas > tsa_threshold, torch.tensor([1], dtype=torch.uint8),
                                    torch.tensor([0],  dtype=torch.uint8))
            loss_mask.to(device)
            loss_mask = loss_mask.view(-1)
            loss[loss_mask] = 0.
            number_of_elements = loss_mask.size(0) - loss_mask.sum(0)
            if verbose:
                print('outputs', softmaxed)
                print('tsa_threshold', tsa_threshold)
                print('label_ids', target)
                print('probas', probas)
                print('mask', loss_mask)
                print('post_loss', loss)
                print('number_of_elements : ', loss_mask.size(0) - loss_mask.sum(0))


            if number_of_elements > 0:
                loss = loss[loss > 0.]
                nb_elements = loss.size(0)
                loss = loss.mean(-1)
                loss.backward()
            else:
                loss = torch.tensor([0.])
                loss = loss.mean(-1)
        else:
            loss = loss.mean(-1)
            loss.backward()

        #loss.backward()
        optimizer.step()
        global_step += 1

    if epoch or epochs:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader),
            loss.data))

    return loss.data

def train_semi(epoch):
    model.train()
    loss = None
    global train_data
    # update global step per epoch
    global_step = (epoch - 1) * (len(train_data) / args.batch_size)
    num_train_step = 50 * len(train_data) / args.batch_size

    for batch_idx, (labeled, unlabeled) in enumerate(zip(train_loader, train_pool_loader)):
        data, target = labeled
        data_unlabeled, _ = unlabeled
        if args.augment:
            data = augment(data)  # augment labeled data
            data_unlabeled = augment(data_unlabeled)

        data, target = to_gpu(Variable(data), args.cuda), to_gpu(Variable(target), args.cuda)
        data_unlabeled = to_gpu(Variable(data_unlabeled), args.cuda)
        optimizer.zero_grad()
        # supervised model
        outputs = model(data)
        softmaxed = F.softmax(outputs, dim=-1)
        # unlabeled version of model on unlabeled data
        outputs_unlabeled = model(data_unlabeled)
        fake_target = Variable(outputs_unlabeled.data.max(1)[1].view(-1))
        criterion_labeled = nn.CrossEntropyLoss(reduction='none')
        criterion = nn.CrossEntropyLoss()
        loss = criterion_labeled(outputs, target)
        # unsupervised / reconstruction loss
        unlabeled_loss = criterion(outputs_unlabeled, fake_target)
        #loss += unlabeled_loss

        if tsa:
            tsa_start = 1./nb_classes
            tsa_threshold = get_tsa_threshold(global_step=global_step, \
                                              num_train_step=num_train_step, start=tsa_start, \
                                              end=1., schedule='linear', scale=5)
            loss_mask = torch.ones(loss.size()).long()
            probas = torch.gather(softmaxed, dim=-1, index=target.unsqueeze(1)).cpu()
            loss_mask = torch.where(probas > tsa_threshold, torch.tensor([1], dtype=torch.uint8),
                                    torch.tensor([0],  dtype=torch.uint8))
            loss_mask.to(device)
            loss_mask = loss_mask.view(-1)
            loss[loss_mask] = 0.
            number_of_elements = loss_mask.size(0) - loss_mask.sum(0)
            if verbose:
                print('outputs', softmaxed)
                print('tsa_threshold', tsa_threshold)
                print('label_ids', target)
                print('probas', probas)
                print('mask', loss_mask)
                print('post_loss', loss)
                print('number_of_elements : ', loss_mask.size(0) - loss_mask.sum(0))

        if number_of_elements > 0:
            loss = loss[loss > 0.]
            nb_elements = loss.size(0)
            loss = loss.mean(-1)
            loss += unlabeled_loss
            loss.backward()
        else:
            nb_elements = 0
            loss = torch.tensor([0.])
            loss += unlabeled_loss

        #loss.backward()
        optimizer.step()
        global_step += 1

    if epoch or epochs:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.data))

    return loss.data

def evaluate(input_data, stochastic = False, predict_classes=False):

    if stochastic:
        model.train() # we use dropout at test time
    else:
        model.eval()
    loss_score = []
    predictions = []
    test_loss = 0
    correct = 0
    for data, target in input_data:
        if args.cuda:
            data, target = data.cuda(device), target.cuda(device)
        data, target = Variable(data, volatile=True), Variable(target)

        output = model(data)
        softmaxed = F.softmax(output.cpu())
        
        if predict_classes:
            predictions.extend(np.argmax(softmaxed.data.numpy(),axis = -1))
        else:
            predictions.extend(softmaxed.data.numpy())
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target)

        fake_target = Variable(output.data.max(1)[1].view(-1))
        criterion_unlabeled = nn.CrossEntropyLoss(reduction='none')
        loss_unlabeled = criterion_unlabeled(output, fake_target)

        loss_score.extend(loss_unlabeled.cpu())
        test_loss += loss.data
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()
    return (test_loss, correct, predictions)

def val(epoch):
    val_loss = 0
    va_correct = 0
    val_loss, val_correct,_ =  evaluate(val_loader, stochastic= False)

    val_loss /= len(val_loader) # loss function already averages over batch size


    if epoch == epochs:
        print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, val_correct, len(val_loader.dataset),
        100. * val_correct / len(val_loader.dataset)))

    return  val_loss, 100. * val_correct / len(val_loader.dataset)

def test(epoch):
    test_loss = 0
    correct = 0
    test_loss, correct,_ =  evaluate(test_loader, stochastic= False)

    test_loss /= len(test_loader) # loss function already averages over batch size
    if epoch or  epochs:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return test_loss, 100. * correct / len(test_loader.dataset)


def acquire_points(argument, random_sample=False):
    global train_data
    global train_target

    acquisition_iterations = 98
    dropout_iterations = 50
    Queries = 20
    pool_all = np.zeros(shape=(1))

    if argument == "RANDOM":
        random_sample = True
    else:
        acquisition_function = getAcquisitionFunction(argument)

    val_loss_hist = []
    val_acc_hist = []
    test_loss_hist = []
    test_acc_hist = []
    train_loss_hist = []
    for i in range(acquisition_iterations):
        pool_subset = 2000
        if random_sample:
            pool_subset = Queries
        print("Acquisition Iteration " + str(i))
        pool_subset_dropout = torch.from_numpy(np.asarray(random.sample(range(0, pool_data.size(0)), pool_subset)))
        pool_data_dropout = pool_data[pool_subset_dropout]
        pool_target_dropout = pool_target[pool_subset_dropout]
        if random_sample is True:
            pool_index = np.array(range(0, Queries))

        else:
            points_of_interest = acquisition_function(dropout_iterations, pool_data_dropout, pool_target_dropout)
            if argument == "UNCERTAINTY":
                # select queries number
                pool_index = points_of_interest.argsort()[-Queries:]
                # select uncertainty low than 0.2
                print(points_of_interest)

                #container = np.arange(len(points_of_interest))
                #pool_index = container[points_of_interest<0.5]
                print(pool_index)
            else:
                pool_index = points_of_interest.argsort()[-Queries:][::-1]

        pool_index = torch.from_numpy(np.flip(pool_index, axis=0).copy())

        pool_all = np.append(pool_all, pool_index)

        pooled_data = pool_data_dropout[pool_index]
        pooled_target = pool_target_dropout[pool_index]
        train_data = torch.cat((train_data, pooled_data), 0)
        train_target = torch.cat((train_target, pooled_target), 0)

        # remove from pool set
        remove_pooled_points(pool_subset, pool_data_dropout, pool_target_dropout, pool_index)

        train_loss, val_loss, test_loss, val_accuracy, test_accuracy = train_test_val_loop(init_train_set=True, disable_test=False)

        val_loss_hist.append(val_loss)
        val_acc_hist.append(val_accuracy)
        test_loss_hist.append(test_loss)
        train_loss_hist.append(train_loss)
        test_acc_hist.append(test_accuracy)

    """np.save("./val_loss_" + argument + ".npy", np.asarray(val_loss_hist))
    np.save("./val_acc_" + argument + ".npy", np.asarray(val_acc_hist))
    np.save("./train_loss_" + argument + ".npy", np.asarray(train_loss_hist))
    np.save("./test_loss_" + argument + ".npy", np.asarray(test_loss_hist))
    np.save("./test_acc_" + argument + ".npy", np.asarray(test_acc_hist))"""


def getAcquisitionFunction(name):
    if name == "BALD":
        return bald_acquisition
    elif name == "VAR_RATIOS":
        return variation_ratios_acquisition
    elif name == "MAX_ENTROPY":
        return max_entroy_acquisition
    elif name == "MEAN_STD":
        return mean_std_acquisition
    elif name == "UNCERTAINTY":
        return uncertainty_acquisition
    elif name=="LOSS":
        return loss_acquisition
    else:
        print ("ACQUSITION FUNCTION NOT IMPLEMENTED")

def loss_acquisition(dropout_iterations, pool_data_dropout, pool_target_dropout):
    print("LOSS FUNCTION")
    score_All = np.zeros(shape=(pool_data_dropout.size(0), 1))
    # Validation Dataset
    pool = Data.TensorDataset(pool_data_dropout, pool_target_dropout)
    pool_loader = Data.DataLoader(pool, batch_size=args.batch_size, shuffle=True)
    start_time = time.time()
    for d in range(dropout_iterations):
        _, _, loss = evaluate(pool_loader, stochastic=True)
        score = np.array(loss)
        score = np.expand_dims(score, axis=1)
        print(score.shape)
        score_All = score_All + score

    print("Dropout Iterations took --- %s seconds ---" % (time.time() - start_time))
    print(score_All.shape)
    points_of_interest = np.divide(score_All, dropout_iterations)
    points_of_interest = points_of_interest.flatten()
    print(points_of_interest.shape)
    return  points_of_interest


def uncertainty_acquisition(dropout_iterations, pool_data_dropout, pool_target_dropout):
    print("UNCERTAINTY FUNCTION")
    score = np.zeros(shape=(pool_data_dropout.size(0), 1, nb_classes))
    # Validation Dataset
    pool = Data.TensorDataset(pool_data_dropout, pool_target_dropout)
    pool_loader = Data.DataLoader(pool, batch_size=args.batch_size, shuffle=True)
    start_time = time.time()
    for d in range(dropout_iterations):
        _, _, predictions = evaluate(pool_loader, stochastic=True)
        predictions = np.array(predictions)
        predictions = np.expand_dims(predictions, axis=1)
        score = np.append(score, predictions, axis=1)

    print("Dropout Iterations took --- %s seconds ---" % (time.time() - start_time))
    max_prob = []
    print(score.shape)
    for i in range(len(pool_data_dropout)):
        all_classes_prob = []
        for j in range(nb_classes):
            histo = []
            for z in range(dropout_iterations):
                histo.append(score[i][z][j])
            #prob = np.percentile(histo, 50)
            prob = np.amax(histo)
            all_classes_prob.append(prob)
        max = np.amax(all_classes_prob)
        max_prob.append(max)

    points_of_interest = np.array(max_prob)
    print(points_of_interest.shape)
    return  points_of_interest

def max_entroy_acquisition(dropout_iterations, pool_data_dropout, pool_target_dropout):
    print("MAX ENTROPY FUNCTION")
    score_All = np.zeros(shape=(pool_data_dropout.size(0), nb_classes))
    # Validation Dataset
    pool = Data.TensorDataset(pool_data_dropout, pool_target_dropout)
    pool_loader = Data.DataLoader(pool, batch_size=args.batch_size, shuffle=True)
    start_time = time.time()
    for d in range(dropout_iterations):
        _, _, predictions = evaluate(pool_loader, stochastic=True)

        predictions = np.array(predictions)
        #predictions = np.expand_dims(predictions, axis=1)
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
    return  points_of_interest


def mean_std_acquisition(dropout_iterations, pool_data_dropout, pool_target_dropout):
    print("MEAN STD ACQUISITION FUNCTION")
    all_dropout_scores = np.zeros(shape=(pool_data_dropout.size(0), 1))
    print(all_dropout_scores)
    # Validation Dataset
    pool = Data.TensorDataset(pool_data_dropout, pool_target_dropout)
    pool_loader = Data.DataLoader(pool, batch_size=args.batch_size, shuffle=True)
    start_time = time.time()
    for d in range(dropout_iterations):
        _, _, scores = evaluate(pool_loader, stochastic=True)

        scores = np.array(scores)
        all_dropout_scores = np.append(all_dropout_scores, scores, axis=1)
    print("Dropout Iterations took --- %s seconds ---" % (time.time() - start_time))
    std_devs= np.zeros(shape = (pool_data_dropout.size(0),nb_classes))
    sigma = np.zeros(shape = (pool_data_dropout.size(0)))
    for t in range(pool_data_dropout.size(0)):
        for r in range( nb_classes ):
            L = np.array([0])
            for k in range(r + 1, all_dropout_scores.shape[1], 10 ):
                L = np.append(L, all_dropout_scores[t, k])

            L_std = np.std(L[1:])
            std_devs[t, r] = L_std
        E = std_devs[t, :]
        sigma[t] = sum(E)/nb_classes


    points_of_interest = sigma.flatten()
    return points_of_interest

def bald_acquisition(dropout_iterations, pool_data_dropout, pool_target_dropout):
    print ("BALD ACQUISITION FUNCTION")
    score_all = np.zeros(shape=(pool_data_dropout.size(0), nb_classes))
    all_entropy = np.zeros(shape=pool_data_dropout.size(0))

    # Validation Dataset
    pool = Data.TensorDataset(pool_data_dropout, pool_target_dropout)
    pool_loader = Data.DataLoader(pool, batch_size=args.batch_size, shuffle=True)
    start_time = time.time()
    for d in range(dropout_iterations):
        _, _, scores = evaluate(pool_loader, stochastic=True)

        scores = np.array(scores)
        #predictions = np.expand_dims(predictions, axis=1)
        score_all = score_all + scores

        log_score = np.log2(scores)
        entropy = - np.multiply(scores, log_score)
        entropy_per_dropout = np.sum(entropy,axis =1)
        all_entropy = all_entropy + entropy_per_dropout


    print("Dropout Iterations took --- %s seconds ---" % (time.time() - start_time))
    # print (All_Dropout_Classes)
    avg_pi = np.divide(score_all, dropout_iterations)
    log_avg_pi = np.log2(avg_pi)
    entropy_avg_pi = - np.multiply(avg_pi, log_avg_pi)
    entropy_average_pi = np.sum(entropy_avg_pi, axis=1)

    g_x = entropy_average_pi
    average_entropy = np.divide(all_entropy,dropout_iterations)
    f_x = average_entropy

    u_x = g_x - f_x


    # THIS FINDS THE MINIMUM INDEX
    # a_1d = U_X.flatten()
    # x_pool_index = a_1d.argsort()[-Queries:]


    points_of_interest = u_x.flatten()
    return  points_of_interest

def variation_ratios_acquisition(dropout_iterations, pool_data_dropout, pool_target_dropout):
    print("VARIATIONAL RATIOS ACQUSITION FUNCTION")
    All_Dropout_Classes = np.zeros(shape=(pool_data_dropout.size(0), 1))
    # Validation Dataset
    pool = Data.TensorDataset(pool_data_dropout, pool_target_dropout)
    pool_loader = Data.DataLoader(pool, batch_size=args.batch_size, shuffle=True)
    start_time = time.time()
    for d in range(dropout_iterations):
        _, _, predictions = evaluate(pool_loader, stochastic=True,predict_classes=True)

        predictions = np.array(predictions)
        predictions = np.expand_dims(predictions, axis=1)
        All_Dropout_Classes = np.append(All_Dropout_Classes, predictions, axis=1)
    print("Dropout Iterations took --- %s seconds ---" % (time.time() - start_time))
    # print (All_Dropout_Classes)
    Variation = np.zeros(shape=(pool_data_dropout.size(0)))
    for t in range(pool_data_dropout.size(0)):
        L = np.array([0])
        for d_iter in range(dropout_iterations):
            L = np.append(L, All_Dropout_Classes[t, d_iter + 1])
        Predicted_Class, Mode = mode(L[1:])
        v = np.array([1 - Mode / float(dropout_iterations)])
        Variation[t] = v
    points_of_interest = Variation.flatten()
    return points_of_interest

def remove_pooled_points(pool_subset, pool_data_dropout, pool_target_dropout, pool_index):
    global pool_data
    global pool_target
    np_data = pool_data.numpy()
    np_target = pool_target.numpy()
    pool_data_dropout = pool_data_dropout.numpy()
    pool_target_dropout = pool_target_dropout.numpy()
    np_index =  pool_index.numpy()
    np.delete(np_data, pool_subset,axis =0)
    np.delete(np_target, pool_subset,axis =0)

    np.delete(pool_data_dropout,np_index,axis = 0)
    np.delete(pool_target_dropout,np_index, axis=0)

    np_data = np.concatenate((np_data,pool_data_dropout),axis =0)
    np_target = np.concatenate((np_target, pool_target_dropout),axis =0)

    pool_data = torch.from_numpy(np_data)
    pool_target = torch.from_numpy(np_target)


def train_test_val_loop(init_train_set, disable_test=True):
    if init_train_set:
        initialize_train_set()
    init_model()

    train_loss = 0
    val_loss = 0
    val_accuracy = 0
    test_loss = -1
    test_accuracy = -1
    print("Training again")
    for epoch in range(1, epochs + 1):
        if args.supervised:
            train_loss = train(epoch)
        else:
            train_loss = train_semi(epoch)
            print("unsupervised learning")
        val_loss, val_accuracy = val(epoch)

    if disable_test is False:
        test_loss, test_accuracy = test(epoch)

    return train_loss, val_loss, test_loss, val_accuracy, test_accuracy


def init_model():
    global model
    global optimizer
    model = Net()
    if args.cuda:
        model.cuda(device)
    decay = 3.5 / train_data.size(0)
    optimizer = optim.Adam([{'params': model.conv.parameters()}, {'params': model.fc.parameters()},
                            {'params': model.dense.parameters(), 'weight_decay': decay}], lr=LR)
    #optimizer = optim.Adam(model.parameters(), weight_decay=decay, lr=LR)
start_time = time.time()
initialize_train_set()
init_model()
print("Training without acquisition")
for epoch in range(1, epochs + 1):
    if args.supervised:
        train_loss = train(epoch)
        print("supervised learning")
    else:
        train_loss = train_semi(epoch)
        print("unsupervised learning")
    val_loss, accuracy = val(epoch)

print("acquring points")
init_model()
acquire_points(args.acqu)
print("Training again")
train_test_val_loop(init_train_set=True, disable_test=False)  # disable_test=False

print("--- %s seconds ---" % (time.time() - start_time))

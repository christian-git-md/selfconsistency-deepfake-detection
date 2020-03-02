from __future__ import division

import argparse
import os
import shutil
import time
import datetime
import json
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from utils import freeze_all, save_models, get_dataset_from_type, get_dict_value_or_none, Triplet_Selector
from network import OnlineFeatureNetwork, ContrastiveLoss, TripletLoss

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100, help='epoch number')
parser.add_argument('--bs', type=int, default=64, help='batch size')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
args = parser.parse_args()
print("training started with settings: {}".format(args))

batch_size = args.bs  # Batch size
lr = args.lr  # learning rate
epochs = args.epochs  # number of epochs
num_per_im = 8  # number of patches cropped from each image
model_save_iter = 1500  # inverval of model saving
margin = 0.2  # value for the margin variable that appears in the triplet loss
embedding_size = 128
load = True  # load a saved model
validation = False  # use validation data while training
num_test_iter = 1
parallel = False


def train_with_pretraining(model, num_epochs):
    """
    Use a pretrained model and do some steps for warmup,
    where the non-head part of the model stays frozen
    """
    model = freeze_all(model=model)
    for param in model.resnet._modules['fc'].parameters():
        param.requires_grad = True
    print('start of pretraining')
    print(num_epochs)
    train(model, num_epochs=num_epochs, is_pretraining=True)
    return


def train_head(model, num_epochs=epochs):
    for param in model._modules['feat_extractor'].parameters():
        param.requires_grad = False
    for param in model._modules['fc'].parameters():
        param.requires_grad = True
        train(model, num_epochs=num_epochs, is_pretraining=False)
    return


def unfreeze(model):
    model = freeze_all(model=model)
    # all parameters are listed here for completeness
    for param in model.resnet._modules['conv1'].parameters():
        param.requires_grad = True
    for param in model.resnet._modules['layer1'].parameters():
        param.requires_grad = True
    for param in model.resnet._modules['layer2'].parameters():
        param.requires_grad = True
    for param in model.resnet._modules['layer3'].parameters():
        param.requires_grad = True
    for param in model.resnet._modules['layer4'].parameters():
        param.requires_grad = True
    for param in model.resnet._modules['avgpool'].parameters():
        param.requires_grad = True
    for param in model.resnet._modules['fc'].parameters():
        param.requires_grad = True
    print('start of main training')
    return model


def test(model):
    count_1 = 0.000001
    count_0 = 0.000001
    tp = 0.
    tn = 0.
    global test_loader_iterator
    model.eval()
    with torch.no_grad():
        for i in range(0, num_test_iter):
            try:
                im_a, im_b, labels = next(test_loader_iterator)
            except StopIteration:
                test_loader_iterator = iter(test_loader)
                im_a, im_b, labels = next(test_loader_iterator)
            im_a = im_a.cuda()
            im_b = im_b.cuda()
            labels = labels.cuda()
            outputs = model(im_a.unsqueeze(0)), model(im_b.unsqueeze(0))
            prediction = loss_fn.predict_tuplet(outputs)
            count_1 += torch.sum((labels == 1), dim=0).float() + 0.00001
            count_0 += torch.sum((labels == 0), dim=0).float() + 0.00001
            tp += torch.sum((prediction == labels) * (labels == 1), dim=0)
            tn += torch.sum((prediction == labels) * (labels == 0), dim=0)
        test_acc, test_exif_acc, test_balanced_acc, test_balanced_exif_acc = getmetrics(count_0, count_1, tp, tn)
        return test_acc, test_exif_acc, test_balanced_acc, test_balanced_exif_acc


def reset_tensorboard():
    try:
        shutil.rmtree(os.getcwd() + '/runs')
    except OSError as e:
        print(e)
    writer = SummaryWriter(log_dir="runs/train" + datetime.datetime.now().strftime("%Y-%m-%d_%H:%M"))
    testwriter = SummaryWriter(log_dir="runs/test" + datetime.datetime.now().strftime("%Y-%m%d_%H:%M"))
    return writer, testwriter


def getmetrics(count_0, count_1, tp, tn):
    balanced_exif_acc = (tp.float() / count_1.float() + tn.float() / count_0.float()) * 0.5
    balanced_acc = torch.mean(balanced_exif_acc)
    acc = torch.sum(tp + tn).data.tolist() / (float(torch.sum(count_1 + count_0)))
    exif_acc = (tp + tn).float() / (count_1 + count_0).float()
    return acc, exif_acc, balanced_acc, balanced_exif_acc


def train(model, num_epochs, is_pretraining=False):
    train_loss = 0.0
    count_0 = 0.000001  # to avoid dividing by something bad
    count_1 = 0.000001
    tp = 0.0
    tn = 0.0

    t = time.time()
    j = 0
    for epoch in range(num_epochs):
        print("Starting Epoch " + str(epoch))
        for im, labels in train_loader:
            model.eval()
            im = im.cuda()
            # Clear all accumulated gradients
            optimizer.zero_grad()
            # Predict classes using images from the test set
            outputs = model(im)
            # Create pairs of right and wrong samples for training metrics
            with torch.no_grad():
                b_size = im.shape[0]
                right = outputs.view(-1, 2, embedding_size)
                wrong = outputs.view(-1, num_per_im, embedding_size).transpose(0, 1).reshape(-1, 2, embedding_size)
                prediction = loss_fn.predict_tuplet(torch.cat((right, wrong), dim=0).transpose(0, 1))
                same_image = torch.cuda.FloatTensor(size=(int(b_size * num_per_im / 2),)).fill_(1)
                diff_image = torch.cuda.FloatTensor(size=(int(b_size * num_per_im / 2),)).fill_(0)
                labels = torch.cat((same_image, diff_image), dim=0).unsqueeze(1)
                count_1 += torch.sum((labels == 1), dim=0).float() + 0.00001
                count_0 += torch.sum((labels == 0), dim=0).float() + 0.00001
                tp += torch.sum((prediction == labels) * (labels == 1), dim=0)
                tn += torch.sum((prediction == labels) * (labels == 0), dim=0)
            triplets = triplet_selector.select_triples(outputs)

            triplet_outputs = outputs.view(-1, embedding_size)[triplets.view(-1)].view(
                triplets.shape[0], triplets.shape[1], embedding_size).transpose_(0, 1)
            with torch.no_grad():
                num_semi_hard = triplets.shape[0]
            del outputs
            loss = loss_fn(triplet_outputs)
            # Backpropagate the loss and adjust parameters
            loss.backward()
            optimizer.step()
            train_loss += loss
            if j % 20 == 19:
                acc, exif_acc, balanced_acc, balanced_exif_acc = getmetrics(count_0, count_1, tp, tn)
                print("Iterations  " + str(j + 1))
                print("Train time   " + str((time.time() - t) / 20))
                writer.add_scalar('asummary/accuracy', acc, j)
                writer.add_scalar('asummary/balanced_accuracy', balanced_acc, j)
                writer.add_scalar('asummary/loss', train_loss.data.tolist() / (20), j)
                writer.add_scalar('asummary/num_semi_hard_samples', num_semi_hard, j)
                if validation:
                    test_acc, test_exif_acc, test_balanced_acc, test_balanced_exif_acc = test(model)
                    test_writer.add_scalar('asummary/accuracy', test_acc, j)
                    test_writer.add_scalar('asummary/balanced_accuracy', test_balanced_acc, j)
                train_loss = 0.
                tp = 0.
                tn = 0.
                count_1 = 0.0000001
                count_0 = 0.0000001
                t = time.time()
            if j == num_pretrain_steps and epoch == 0 and is_pretraining:
                model = unfreeze(model)
                if parallel:
                    model = torch.nn.DataParallel(model)
            if j % model_save_iter == 0:
                save_models(j, model)
            j += 1
    print("training finished with settings: {}. " "best testing accuracy was: {}".format(args, acc))
    return model


if __name__ == "__main__":
    triplet_selector = Triplet_Selector(num_per_im=num_per_im)
    config_file_path = "config/preextracted_triplet.json"
    with open(config_file_path) as fp:
        config = json.load(fp)
    num_train_samples = get_dict_value_or_none(config, 'num_train_samples')
    num_test_samples = get_dict_value_or_none(config, 'num_test_samples')

    num_pretrain_steps = config['warm_up_iterations']
    taglist = ['image']
    train_dataset = get_dataset_from_type(config['train_loader'], config["train_path"], num_train_samples,
                                          preprocess=config['augmentations'])
    test_dataset = get_dataset_from_type(config['test_loader'], config["test_path"], num_test_samples,
                                         preprocess=config['augmentations'])
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size * num_per_im * 32)
    test_loader_iterator = iter(test_loader)
    writer, test_writer = reset_tensorboard()
    # Create model, optimizer and loss function
    mod = OnlineFeatureNetwork()
    mod.cuda()
    optimizer = Adam(mod.parameters(), lr=lr)
    loss_fn = TripletLoss(margin=margin)
    test_loss_fn = ContrastiveLoss(margin=margin)
    if load and config['model_load_path']:
        mod.load_state_dict(torch.load(config['model_load_path']))
        train(mod, num_epochs=epochs)
    # freezes part of the model for pretraining and then another part of the model for main training (transfer learning)
    elif num_pretrain_steps:
        train_with_pretraining(mod, num_epochs=epochs)
    else:
        train(mod, num_epochs=epochs)

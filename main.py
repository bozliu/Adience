import os
import argparse
import warnings

from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

from data import Facedata
from models import Resnet, VGG, Base_CNN, LabelSmoothingCrossEntropy
from utils import *

warnings.filterwarnings("ignore")


# ---------------------------Parameters----------------------------
parser = argparse.ArgumentParser(description='PyTorch Gender_age_detection Training')
parser.add_argument('-d', '--data', default='./Adience_Benchmark', type=str)
parser.add_argument('--seed', default=2020, type=int)
parser.add_argument('--model', default='resnet50', type=str, help='resnet18, resnet34, resnet50, VGG16, VGG19, CNNbase')
parser.add_argument('--epochs', default=25, type=int, metavar='N')
parser.add_argument('-b', '--batchsize', default=32, type=int)
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float)
parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained resnet')
parser.add_argument('--gender-weight', default=1, type=float)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--drop-out', default=0.5, type=float)
parser.add_argument('-c', '--checkpoint', default='./logs/checkpoints', type=str, metavar='PATH', help='path to save checkpoint (default: checkpoints)')
parser.add_argument('--resume-checkpoint', default='', type=str, metavar='PATH', help='resume training from checkpoint')
parser.add_argument('--scheduler', default='step', type=str)
parser.add_argument('--optimizer', default='Adam', type=str, help='[Adam, SGD]')
parser.add_argument('--loss', default='CE', type=str, help='[CE, smoothCE]')
parser.add_argument('--orderloss', default=0.01, type=float, help='order loss for age')

args = parser.parse_args()


def main(fold=0):
    args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    seed_everything(args.seed)
    check_print_args(args)
    start_epoch, best_age_acc, best_gender_acc = 0, 0, 0
    train_loss_list, val_loss_list, train_gender_list, val_gender_list, train_age_list, val_age_list = [], [], [], [], [], []
    if not os.path.isdir(args.checkpoint):
        os.makedirs(args.checkpoint)

    # whether using pretrained model?
    if args.model == 'resnet50':
        model = Resnet(layers=50, pretrained=args.pretrained, drop_rate=args.drop_out)
    if args.model == 'resnet34':
        model = Resnet(layers=34, pretrained=args.pretrained, drop_rate=args.drop_out)
    if args.model == 'resnet18':
        model = Resnet(layers=18, pretrained=args.pretrained, drop_rate=args.drop_out)
    if args.model == 'VGG16':
        model = VGG(layers=16, pretrained=args.pretrained, drop_rate=args.drop_out)
    if args.model == 'VGG19':
        model = VGG(layers=19, pretrained=args.pretrained, drop_rate=args.drop_out)
    if args.model == 'CNNbase':
        model = Base_CNN(drop_rate=args.drop_out)

    model = model.to(args.device)
    # model.fine_tune(args.fine_tune) # fine_tune the model? (only for resnet)

    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), args.lr, weight_decay=1e-4)
    elif args.optimizer == "SGD":
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), args.lr, momentum=0.9, weight_decay=1e-4)

    if args.loss == "CE":
        criterion = nn.CrossEntropyLoss().to(args.device)
    elif args.loss == "smoothCE":
        criterion = LabelSmoothingCrossEntropy().to(args.device)

    if args.scheduler not in ['step', 'ReduceLROnPlateau', 'CosineAnnealing']:
        raise ValueError('scheduler must in (step, ReduceLROnPlateau, CosineAnnealing)')
    else:
        if args.scheduler == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.25)
        elif args.scheduler == 'ReduceLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
        elif args.scheduler == 'CosineAnnealing':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 5, eta_min=1e-8)

    # loading checkpoint
    if args.resume_checkpoint:
        if os.path.isfile(args.resume_checkpoint):
            print("=> loading checkpoint '{}'".format(args.resume_checkpoint))
            checkpoint = torch.load(args.resume_checkpoint, map_location=str(args.device))
            start_epoch = checkpoint['epoch']
            best_age_acc = checkpoint['best_age_acc']
            best_gender_acc = checkpoint['best_gender_acc']
            train_loss_list = checkpoint['train_loss_list']
            val_loss_list = checkpoint['val_loss_list']
            train_gender_list = checkpoint['train_gender_list']
            val_gender_list = checkpoint['val_gender_list']
            train_age_list = checkpoint['train_age_list']
            val_age_list = checkpoint['val_age_list']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {} best_age_acc {} best_gender_acc {})".format(args.resume_checkpoint, start_epoch, best_age_acc, best_gender_acc))
            args.checkpoint = os.path.dirname(args.resume_checkpoint)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume_checkpoint))

    print('loading image....')

    # loading the dataset.....
    split = [j for j in range(5) if j != fold]
    train_dataset = Facedata(data_path=args.data, folds=[j for j in split if j != (fold-1) % 5])
    val_dataset = Facedata(data_path=args.data, folds=[(fold-1) % 5], img_augment=None)
    test_dataset = Facedata(data_path=args.data, folds=[fold], img_augment=None) 
    train_loader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batchsize, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batchsize, shuffle=False)

    print('loading successful')
    print('start training')

    # the training part...
    for epoch in range(start_epoch, args.epochs):

        print(f"Epoch: {epoch + 1}|{args.epochs},  LR: {optimizer.state_dict()['param_groups'][0]['lr']}")

        train_loss, train_gender_acc, train_age_acc = train(train_loader, model, optimizer, epoch, scheduler, criterion)
        val_loss, val_gender_acc, val_age_acc, _ = validate(val_loader, model, criterion)

        # statistics
        train_loss_list.append(train_loss); val_loss_list.append(val_loss); train_gender_list.append(train_gender_acc)
        val_gender_list.append(val_gender_acc); train_age_list.append(train_age_acc); val_age_list.append(val_age_acc)
        is_best = val_age_acc > best_age_acc and val_gender_acc > best_gender_acc
        if is_best:
            best_age_acc = max(val_age_acc, best_age_acc)
            best_gender_acc = max(val_gender_acc, best_gender_acc)

        print(f'>>Epoch {epoch+1}/{args.epochs}, LR: {optimizer.state_dict()["param_groups"][0]["lr"]}, train loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, train_gender_acc:{train_gender_acc*100:.4f}%, train_age_acc:{train_age_acc*100:.4f}%, val_gender_acc:{val_gender_acc*100:.4f}%, val_age_acc:{val_age_acc*100:.4f}%')

        if args.scheduler == 'step':
            scheduler.step()
        elif args.scheduler == 'ReduceLROnPlateau':
            scheduler.step(val_loss)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_age_acc': best_age_acc,
            'best_gender_acc': best_gender_acc,
            'optimizer': optimizer.state_dict(),
            'train_loss_list': train_loss_list,
            'val_loss_list': val_loss_list,
            'train_gender_list': train_gender_list,
            'val_gender_list': val_gender_list,
            'train_age_list': train_age_list,
            'val_age_list': val_age_list,
        }, is_best, checkpoint=args.checkpoint, filename=f'checkpoint_{args.model}_FOLD{fold}.pth.tar')

    # plot(train_loss_list, val_loss_list, train_gender_list, val_gender_list, train_age_list, val_age_list, fold=fold, path=args.checkpoint)
    print(f">> Fold{fold}: best_gender_acc: {best_gender_acc}, best_age_acc: {best_age_acc}")

    # use best_model to inference on val set
    pre = os.path.join(args.checkpoint, f"Best_checkpoint_{args.model}_FOLD{fold}")
    path = pre+'.pth.tar'
    checkpoint = torch.load(path, map_location=str(args.device))
    model.load_state_dict(checkpoint['state_dict'])
    _, _, _, res = validate(test_loader, model, criterion)

    return res


def train(train_loader, model, optimizer, epoch, scheduler, criterion):
    model.train()
    train_bs_number = len(iter(train_loader))
    total_loss = 0
    total_instance = 0
    right_gender = 0
    right_age = 0

    for i, (sample, gender, age) in enumerate(train_loader):
        if i % 50 == 0:
            print(f'current batch  -------  {i + 1}|{train_bs_number}, LR: {optimizer.state_dict()["param_groups"][0]["lr"]}')
        sample = sample.to(args.device)
        gender = gender.to(args.device)
        age = age.to(args.device)
        pred_gender, pred_age = model(sample)  # pred_age:(bs, 8)  pred_gender(bs, 2)

        # loss calculating
        loss_gender = criterion(pred_gender, gender)
        loss_age = criterion(pred_age, age)
        # age order loss
        age_proba = F.softmax(pred_age, dim=1) # (bs, 8)
        age_order = age.float().unsqueeze(1).expand(-1,8) # (bs, 8)
        order_diff = torch.abs(age_order - torch.arange(8).to(args.device).expand_as(age_order)) # (bs, 8)
        loss_order = (order_diff * age_proba).sum()

        loss = loss_age + args.gender_weight * loss_gender + args.orderloss * loss_order

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if args.scheduler == 'CosineAnnealing':
            scheduler.step(epoch + i / train_bs_number)

        total_instance += gender.size(0)
        total_loss += loss.detach() * gender.size(0)
        right_gender += float((gender == torch.argmax(pred_gender, dim=1)).detach().sum())
        right_age += float((age == torch.argmax(pred_age, dim=1)).detach().sum())

    mean_loss = total_loss / total_instance
    gender_accuracy = right_gender / total_instance
    age_accuracy = right_age / total_instance

    return float(mean_loss), float(gender_accuracy), float(age_accuracy)


def validate(val_loader, model, criterion):
    model.eval()
    total_loss = 0
    total_instance = 0
    right_gender = 0
    right_age = 0

    for i, (sample, gender, age) in enumerate(val_loader):
        sample = sample.to(args.device)
        gender = gender.to(args.device)
        age = age.to(args.device)
        pred_gender, pred_age = model(sample)  # pred_age:(bs, 8)  pred_gender(bs, 2)

        loss_gender = criterion(pred_gender, gender)
        loss_age = criterion(pred_age, age)
        loss = loss_age + args.gender_weight * loss_gender

        total_instance += gender.size(0)
        total_loss += loss.detach() * gender.size(0)
        right_gender += float((gender == torch.argmax(pred_gender, dim=1)).detach().sum())
        right_age += float((age == torch.argmax(pred_age, dim=1)).detach().sum())

    mean_loss = total_loss / total_instance
    gender_accuracy = right_gender / total_instance
    age_accuracy = right_age / total_instance

    res = [right_gender, right_age, total_instance]

    return float(mean_loss), float(gender_accuracy), float(age_accuracy), res


if __name__ == '__main__':
    gender, age, instance = 0, 0, 0
    for i in range(5):
        right_gender, right_age, total_instance = main(fold=i)
        gender += right_gender
        age += right_age
        instance += total_instance
    print(f"[!] 5-FOLD CV: Gender Acc: {gender/instance*100:.4f}%, Age ACC: {age/instance*100:.4f}%")

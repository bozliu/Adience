import os
import random
import shutil

import torch
import numpy as np
import matplotlib.pyplot as plt


def seed_everything(seed=2020):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def check_print_args(args):
    assert args.optimizer in ["Adam", "SGD"]
    assert args.loss in ["CE", "smoothCE"]

    print(">> Current Setting:")
    print(f">> Seed: {args.seed}")
    print(f">> Dataset Path: {args.data}")
    print(f'>> Using {args.model} structure {("WITHOUT" if not args.pretrained else "WITH")} pretrained weight')
    print(f">> Training Epoch: {args.epochs}")
    print(f">> Batch Size: {args.batchsize}")
    print(f">> Initial Learning Rate: {args.lr}")
    print(f">> Gender Loss Weight: {args.gender_weight}")
    print(f'>> Device: {args.device}')
    print(f">> Drop Out: {args.drop_out}")
    print(f">> Scheduler: {args.scheduler}")
    print(f">> Optimizer: {args.optimizer}")
    print(f">> Loss: {args.loss}")
    print(f'>> order Loss Weight: {args.orderloss}')
    print(f">> Model Saving Path: {args.checkpoint}")
    if args.resume_checkpoint:
        print(f">> Resume training from: {args.resume_checkpoint}")


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    """
    checkpoint_FOLD{fold}.pth.tar
    Best_checkpoint_FOLD{fold}.pth.tar
    """
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, f'Best_{filename}'))


def show_img(X):
    if X.dim() == 3 and X.size(0) == 3:
        plt.imshow(np.transpose(X.numpy(), (1, 2, 0)))
        plt.show()
    elif X.dim() == 2:
        plt.imshow(X.numpy(), cmap='gray')
        plt.show()
    else:
        print('WRONG TENSOR SIZE')


def show(x, y, label, title, xdes, ydes, path):
    # plt.style.use('ggplot')
    plt.figure(figsize=(16, 8))
    colors = ['tab:green', 'tab:orange', 'tab:blue', 'tab:red', 'tab:pink',
              'tab:purple', 'tab:brown', 'tab:gray', 'tab:olive', 'tab:cyan']
    # plt.xticks(rotation=45)

    assert len(x) == len(y) == len(label)
    for i in range(len(x)):
        plt.plot(x[i], y[i], color=colors[i], label=label[i])
        # marker="o", linestyle='dashed'

    plt.xlabel(xdes)
    # plt.xlim(0, 30000000)
    plt.gca().get_xaxis().get_major_formatter().set_scientific(False)
    plt.gca().get_yaxis().get_major_formatter().set_scientific(False)
    plt.ylabel(ydes)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(path)
    plt.close("all")


def plot(train_loss_list, val_loss_list, train_gender_list, val_gender_list, train_age_list, val_age_list, fold=0, path="./checkpoint"):
    print(f"train_loss_list: {train_loss_list}")
    print(f"train_gender_list: {train_gender_list}")
    print(f"train_age_list: {train_age_list}")
    print(f"val_loss_list: {val_loss_list}")
    print(f"val_gender_list: {val_gender_list}")
    print(f"val_age_list: {val_age_list}")

    epochs = [i+1 for i in range(len(train_loss_list))]

    show(x=[epochs, epochs], y=[train_loss_list, val_loss_list], label=["Train Loss", "Val Loss"],
         title="Gender_Age_Detection", xdes="Epoch", ydes="Loss", path=os.path.join(path, f"loss_{fold}.png"))
    show(x=[epochs, epochs], y=[train_gender_list, val_gender_list], label=["Train Gender Acc", "Val Gender Acc"],
         title="Gender_Age_Detection", xdes="Epoch", ydes="Acc", path=os.path.join(path, "gender_acc_{fold}.png"))
    show(x=[epochs, epochs], y=[train_age_list, val_age_list], label=["Train Age Acc", "Val Age Acc"],
         title="Gender_Age_Detection", xdes="Epoch", ydes="Acc", path=os.path.join(path, "age_acc_{fold}.png"))

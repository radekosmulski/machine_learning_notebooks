from datetime import datetime
import re
import os
import shutil
from subprocess import run
from glob import glob
import torch
import torchvision
import PIL
import numpy as np

def get_fastai_version():
    with open('fastai/../setup.py') as f:
        read_data = f.read()
        return re.search('version = (\d*.\d*)', read_data).groups(0)[0]

def print_info():
    print(f'Last run on: {datetime.now().date()}')
    print(f'PyTorch version: {torch.__version__}')
    print(f'fastai version: {get_fastai_version()}')

def get_mnist(path):
    path = os.path.join(path, 'mnist')

    if not os.path.exists(os.path.join(path, 'processed')):
        torchvision.datasets.MNIST(path, download=True)

    for ds in ['train', 'test']:
        shutil.rmtree(os.path.join(path, ds), ignore_errors=True)
        for i in range(10): os.makedirs(os.path.join(path, ds, str(i)))

    train = torch.load(os.path.join(path, 'processed/training.pt'))
    test = torch.load(os.path.join(path, 'processed/test.pt'))

    np.save(os.path.join(path, 'train_x'), train[0].numpy())
    np.save(os.path.join(path, 'train_y'), train[1].numpy())

    np.save(os.path.join(path, 'test_x'), test[0].numpy())
    np.save(os.path.join(path, 'test_y'), test[1].numpy())

    for i in range(len(train[0])):
        img = PIL.Image.fromarray(train[0][i].numpy())
        label = str(train[1][i])
        img.save(f'{os.path.join(path, "train", label, str(i))}.png', mode='L')

    for i in range(len(test[0])):
        img = PIL.Image.fromarray(test[0][i].numpy())
        label = str(test[1][i])
        img.save(f'{os.path.join(path, "test", label, str(i))}.png', model='L')


def get_cifar10(path):
    shutil.rmtree(f'{path}cifar10')

    run(f'mkdir -p {path}'.split())
    if not os.path.isfile(os.path.join(path, 'cifar.tgz')):
        run(f'wget http://pjreddie.com/media/files/cifar.tgz'.split(), cwd=path)
    if not os.path.isdir(f'{path}cifar'):
        run(f'tar -xf cifar.tgz'.split(), cwd=path)

    for ds in ['train', 'test']:
        trn_paths = glob(f'{path}cifar/{ds}/*')
        for cls in ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'):
            run(f'mkdir -p {path}/cifar10/{ds}/{cls}'.split())
        for fpath in trn_paths:
            cls = re.search('_(.*)\.png$', fpath).group(1)
            fname = re.search('\w*.png$', fpath).group(0)
            shutil.copy(fpath, f'{path}cifar10/{ds}/{cls}/{fname}')


def get_from_dict(d, n):
    return dict(list(d.items())[:n])

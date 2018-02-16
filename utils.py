from datetime import datetime
import torch
import re
import os
import shutil
from subprocess import run
from glob import glob
from IPython.core.debugger import set_trace

def get_fastai_version():
    with open('fastai/../setup.py') as f:
        read_data = f.read()
        return re.search('version = (\d*.\d*)', read_data).groups(0)[0]

def print_info():
    print(f'Last run on: {datetime.now().date()}')
    print(f'PyTorch version: {torch.__version__}')
    print(f'fastai version: {get_fastai_version()}')

def get_cifar10(path):
    if not os.path.isdir(f'{path}cifar10'):
        run(f'mkdir -p {path}'.split())
        run(f'wget http://pjreddie.com/media/files/cifar.tgz'.split(), cwd=path)
        run(f'tar -xf cifar.tgz'.split(), cwd=path)

        for ds in ['train', 'test']:
            trn_paths = glob(f'{path}cifar/{ds}/*')
            for cls in ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'):
                run(f'mkdir -p {path}/cifar10/{ds}/{cls}'.split())
            for fpath in trn_paths:
                cls = re.search('_(.*)\.png$', fpath).group(1)
                fname = re.search('\w*.png$', fpath).group(0)
                os.rename(fpath, f'{path}cifar10/{ds}/{cls}/{fname}')

        shutil.rmtree(f'{path}cifar')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.datasets import ImageFolder
from Pix2pix import Pix2pix
from model import Generator, Discriminator
from dataload import DatasetPipeline
from utils import *
import os


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Current Device: ", device)

    # Set args
    batch_size = 10
    max_epoch = 200
    lr = 2e-4
    trial = 2
    current_epoch = 0
    checkpoint_dir = f'./checkpoint/cp_{trial}_{current_epoch}.pt'
    result_dir = f'./result/cp_{trial}'
    data_root = '../../Datasets/CMPFacade/facade'

    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    # Define Generator and Discriminator
    gen = Generator(3, 3).to(device)
    disc = Discriminator(3).to(device)

    if os.path.exists(checkpoint_dir):
        state_dict = torch.load(checkpoint_dir)
        gen.load_state_dict(state_dict['gen'])
        disc.load_state_dict(state_dict['disc'])
        print("Checkpoint Loaded")
    else:
        gen.apply(weights_init)
        disc.apply(weights_init)
        print("New Model")

    
    # Define Transform
    tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.Resize([256, 256])
            ])

    # Define Dataloder
    dataset = DatasetPipeline(root=data_root)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            drop_last=True,
                            num_workers=4)


    gan_trainer = Pix2pix(gen=gen,
                      disc=disc,
                      max_epoch=max_epoch,
                      batch_size=batch_size,
                      lr=lr,
                      dataloader=dataloader,
                      trial=trial,
                      current_epoch=current_epoch,
                      tf=tf,
                      device=device
                      )
    
    gan_trainer.train()



if __name__=='__main__':
    main()



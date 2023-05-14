import torch
import torch.nn as nn
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from torch.autograd import Variable
from utils import get_result_images


class Pix2pix():
    def __init__(self, 
                 gen, 
                 disc, 
                 max_epoch,
                 batch_size,
                 lr,
                 dataloader,
                 trial,
                 current_epoch,
                 tf,
                 device
                 ):
        
        self.gen = gen
        self.disc = disc
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.lr = lr
        self.dataloader = dataloader
        self.trial = trial
        self.current_epoch= current_epoch
        self.tf = tf
        self.device = device
        self.checkpoint_save_dir = './checkpoint'
        self.lambda_pixel = 10

        self.g_optim = torch.optim.Adam(self.gen.parameters(), lr=lr, betas=(0.5, 0.999))
        self.d_optim = torch.optim.Adam(self.disc.parameters(), lr=lr, betas=(0.5, 0.999))

        self.loss_func = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

        self.real_label = torch.ones((self.batch_size, 1, 16, 16), device=self.device)
        self.fake_label = torch.zeros((self.batch_size, 1, 16, 16), device=self.device)


    def train(self):
        for epoch in range(1, self.max_epoch+1):
            if self.current_epoch >= epoch:
                continue
            self.gen.train()
            self.disc.train()
            for idx, (img, label) in enumerate(tqdm(self.dataloader)):
                # Real Data
                real_data = Variable(img, requires_grad=True).to(self.device)
                label = Variable(label, requires_grad=True).to(self.device)

                # ===== Train Discriminator =====
                self.disc.zero_grad()
                
                fake_data = self.gen(label)
                
                real_pred = self.disc(real_data, label)
                fake_pred = self.disc(fake_data, label)

                real_loss = self.loss_func(real_pred, self.real_label)
                fake_loss = self.loss_func(fake_pred, self.fake_label)

                d_loss = (real_loss + fake_loss) / 2.

                d_loss.backward()
                self.d_optim.step()

                # ===== Train Generator =====
                self.gen.zero_grad()

                fake_data = self.gen(label)

                fake_pred = self.disc(fake_data, label)

                gen_loss = self.loss_func(fake_pred, self.real_label)
                pixel_loss = self.l1_loss(fake_data, real_data)
                
                g_loss = gen_loss + pixel_loss * self.lambda_pixel

                g_loss.backward()
                self.g_optim.step()

                print(f'Epoch {epoch}/{self.max_epoch}, G_Loss: {g_loss.data}, D_Loss: {d_loss.data}')

            torch.save({
                'gen': self.gen.state_dict(),
                'disc': self.disc.state_dict()
            }, f'{self.checkpoint_save_dir}/cp_{self.trial}_{epoch}.pt')

            result = get_result_images(self.gen, self.tf)
            grid = make_grid(result)
            save_image(grid, f'./result/cp_{self.trial}/{str(epoch).zfill(3)}.jpg')


                
        







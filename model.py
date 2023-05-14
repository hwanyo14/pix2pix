import torch
import torch.nn as nn


class DownBlock(nn.Module):
    def __init__(self, chnum_in, chnum_out, norm=True, dropout=0.0) -> None:
        super().__init__()
        layers = [nn.Conv2d(chnum_in, chnum_out, 4, 2, 1, bias=False)]

        if norm:
            layers.append(nn.BatchNorm2d(chnum_out))
        layers.append(nn.LeakyReLU(0.2))

        if dropout:
            layers.append(nn.Dropout(dropout))

        self.down = nn.Sequential(*layers)

    def forward(self, x):
        return self.down(x)


class UpBlock(nn.Module):
    def __init__(self, chnum_in, chnum_out, norm=True, dropout=0.0) -> None:
        super().__init__()
        layers = [nn.ConvTranspose2d(chnum_in, chnum_out, 4, 2, 1, bias=False)]

        if norm:
            layers.append(nn.BatchNorm2d(chnum_out))
        layers.append(nn.LeakyReLU(0.2))

        if dropout:
            layers.append(nn.Dropout(dropout))

        self.up = nn.Sequential(*layers)

    def forward(self, x, y):
        x = self.up(x)
        out = torch.cat([x, y], 1)
        return out
    

class Generator(nn.Module):
    def __init__(self, chnum_in, chnum_out) -> None:
        super().__init__()

        # Encoder
        self.down1 = DownBlock(chnum_in, 64, norm=False)
        self.down2 = DownBlock(64, 128)
        self.down3 = DownBlock(128, 256)
        self.down4 = DownBlock(256, 512)
        self.down5 = DownBlock(512, 512)
        self.down6 = DownBlock(512, 512)
        self.down7 = DownBlock(512, 512)
        self.down8 = DownBlock(512, 512)

        # Decoder
        self.up1 = UpBlock(512, 512, norm=False, dropout=0.5)
        self.up2 = UpBlock(1024, 512, norm=False, dropout=0.5)
        self.up3 = UpBlock(1024, 512, norm=False, dropout=0.5)
        self.up4 = UpBlock(1024, 512, norm=False, )
        self.up5 = UpBlock(1024, 256, norm=False, )
        self.up6 = UpBlock(512, 128, norm=False, )
        self.up7 = UpBlock(256, 64, norm=False, )
        self.up8 = nn.Sequential(
            nn.ConvTranspose2d(128, chnum_out, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        
    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)
        out = self.up8(u7)

        return out


class Discriminator(nn.Module):
    def __init__(self, chnum_in) -> None:
        super().__init__()
        self.disc1 = DownBlock(chnum_in*2, 64, norm=False)
        self.disc2 = DownBlock(64, 128)
        self.disc3 = DownBlock(128, 256)
        self.disc4 = DownBlock(256, 512)
        self.conv1x1 = nn.Conv2d(512, 1, 1, 1, 0, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x, y):
        x = torch.cat([x, y], 1)
        out = self.disc1(x)
        out = self.disc2(out)
        out = self.disc3(out)
        out = self.disc4(out)
        out = self.conv1x1(out)

        return self.sig(out)








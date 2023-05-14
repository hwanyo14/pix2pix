import torch
from torch.utils.data import Dataset
from torchvision import transforms
from glob import glob
from PIL import Image


class DatasetPipeline(Dataset):
    def __init__(self, root) -> None:
        '''
            For CMPFacade dataset
            real image data extension: jpg
            segmentation label data extension: png
        '''
        super().__init__()
        self.root = root
        self.all_img = sorted(glob(f'{root}/*.jpg'))
        self.all_label = sorted(glob(f'{root}/*.png'))

        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.Resize([256, 256])
        ])

    def __getitem__(self, index):
        img = Image.open(self.all_img[index]).convert('RGB')
        label = Image.open(self.all_label[index]).convert('RGB')

        img = self.tf(img)
        label = self.tf(label)

        return img, label
    

    def __len__(self):
        return len(self.all_img)









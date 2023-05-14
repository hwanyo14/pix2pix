import torch
from torch import nn
from PIL import Image



# Weight Initialization
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def get_result_images(gen, tf):
    test_label = '../../Datasets/CMPFacade/test/01.png'
    label = Image.open(test_label).convert("RGB")
    label = tf(label).cuda()
    label = torch.unsqueeze(label, 0)

    gen.eval()

    with torch.no_grad():
        generated = gen(label)

    generated = generated * 0.5 + 0.5

    return generated
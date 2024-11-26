import torch.nn as nn
import torchvision.models as pretrained_models
import torch
from einops import rearrange

from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

vgg_mean = [0.485, 0.456, 0.406]
vgg_std = [0.229, 0.224, 0.225]


class Mask_layers(nn.Module):
    def __init__(self, vgg, method='simple'):
        super(Mask_layers, self).__init__()
        self.method = method
        self.model = torch.nn.Sequential()
        if self.method == 'simple':
            for name, child in vgg.named_children():
                if isinstance(child, nn.MaxPool2d) or isinstance(child, nn.AvgPool2d):
                    self.model.add_module(name, nn.AvgPool2d(kernel_size=2, stride=2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def get_Mask_Models(q_layers):
    return [Mask_layers(q_layers[i]).eval() for i in range(len(q_layers))]


class vgg_Normalization(nn.Module):
    def __init__(self, mean, std, device):
        super().__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).to(device).view(-1, 1, 1)
        self.std = torch.tensor(std).to(device).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


class VGG19(nn.Module):
    def __init__(self, device, if_norm=False):
        super(VGG19, self).__init__()

        # load the vgg model's features
        if if_norm:
            self.normalization = vgg_Normalization(vgg_mean, vgg_std, device).to(device)
        #self.vgg = pretrained_models.vgg19(pretrained=True).features
        self.vgg = pretrained_models.vgg19(weights="IMAGENET1K_V1").features
        self.device = device
        self.if_norm = if_norm

    def get_content_layer(self):
        return [self.vgg[:7]]

    def get_style_layers(self):
        return [self.vgg[:4]] + [self.vgg[:7]] + [self.vgg[:12]] + [self.vgg[:21]] + [self.vgg[:30]]

    def get_content_activations(self, x: torch.Tensor) \
            -> torch.Tensor:
        """
            Extracts the features for the content loss from the block4_conv2 of VGG19
            Args:
                x: torch.Tensor - input image we want to extract the features of
            Returns:
                features: torch.Tensor - the activation maps of the block2_conv1 layer
        """
        y = x
        if self.if_norm:
            y = self.normalization(x)
        features = self.vgg[:7](y)
        return features

    def get_style_activations(self, x):
        """
            Extracts the features for the style loss from the block1_conv1,
                block2_conv1, block3_conv1, block4_conv1, block5_conv1 of VGG19
            Args:
                x: torch.Tensor - input image we want to extract the features of
            Returns:
                features: list - the list of activation maps of the block1_conv1,
                    block2_conv1, block3_conv1, block4_conv1, block5_conv1 layers
        """
        y = x
        if self.if_norm:
            y = self.normalization(x)
        features = [self.vgg[:4](y)] + [self.vgg[:7](y)] + [self.vgg[:12](y)] + [self.vgg[:21](y)] + [self.vgg[:30](y)]
        return features

    def get_sim_activations(self, x):
        y = x
        if self.if_norm:
            y = self.normalization(x)
        features = [self.vgg[:4](y)] + [self.vgg[:7](y)] + [self.vgg[:12](y)] + [self.vgg[:21](y)] + [self.vgg[:30](y)]
        return features

    def forward(self, x):
        y = x
        if self.if_norm:
            y = self.normalization(x)
        return self.vgg(y)


class VGGFeat_Loss(nn.Module):
    def __init__(self, device, if_norm=True):
        super().__init__()
        self.device = device
        self.vgg = VGG19(device, if_norm).to(device)
        # replace the MaxPool with the AvgPool layers
        for name, child in self.vgg.vgg.named_children():
            if isinstance(child, nn.MaxPool2d):
                self.vgg.vgg[int(name)] = nn.AvgPool2d(kernel_size=2, stride=2)

        self.L1Loss = nn.L1Loss().to(device)

        # get mask operation layers
        # self.contentMask_net = get_Mask_Models(self.vgg.get_content_layer())
        # self.styleMask_net = get_Mask_Models(self.vgg.get_style_layers())

        # lock the gradients
        for param in self.vgg.parameters():
            param.requires_grad = False

    def cosin_similarity(self, x, y):
        x, y = map(lambda t: rearrange(t, 'b c h w -> (b h w) c'), (x, y))
        x, y = map(lambda t: t/torch.norm(t, dim=-1).unsqueeze(-1), (x, y))
        xy = torch.mean(torch.sum(x*y, dim=-1))
        score = 1.0 - xy  # 0 ~ 2, 0 best
        return score

    def Gram_distance(self, x, y):
        b, c, h, w = x.shape
        x, y = map(lambda t: rearrange(t, 'b c h w -> b c (h w)'), (x, y))
        gx, gy = map(lambda t: torch.matmul(t, t.transpose(-1, -2)), (x, y))
        dist = torch.sum(torch.pow(gx-gy, 2), dim=(-1, -2))
        dist = dist / (h * w)
        return dist

    def forward(self, X, GX, bsize, method='cos'):
        feat = self.vgg.get_sim_activations(torch.cat([X, GX], dim=0))
        sloss = 0.
        for f in feat:
            if method == 'gram':
                sloss = sloss + self.Gram_distance(f[0:bsize, ...], f[bsize:, ...])
            elif method == 'cos':
                sloss = sloss + self.cosin_similarity(f[0:bsize, ...], f[bsize:, ...])
            else:
                sloss = sloss + self.L1Loss(f[0:bsize, ...], f[bsize:, ...])
        return sloss


if __name__ == '__main__':
    USE_CUDA = torch.cuda.is_available()
    print(USE_CUDA)
    device = torch.device("cuda" if USE_CUDA else "cpu")
    transform = transforms.Compose([transforms.Resize((80, 80)),
                                    transforms.ToTensor()])

    img = Image.open('./test/gt_n.png')
    img = transform(img).to(device)
    img = img.unsqueeze(0)

    img2 = Image.open('./test/c_n.png')
    img2 = transform(img2).to(device)
    img2 = img2.unsqueeze(0)

    VGG_Similarity = VGGFeat_Loss(device).to(device)
    sloss = VGG_Similarity(img, img2, 1, method='cos')
    print(sloss)




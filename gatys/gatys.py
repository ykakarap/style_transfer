# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.utils as utils

import copy

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
# desired size of the output image
# imsize = 512 if torch.cuda.is_available() else 128  # use small size if no gpu
imsize = 512

loader = transforms.Compose([
    transforms.Resize((imsize, imsize)),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor


def image_loader(image_name):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    # unsqueeze to add the "batch" dimension
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

# %%
# style_img = image_loader("images/picasso.jpg")
# content_img = image_loader("images/dancing.jpg")


style_img = image_loader("images/gatys/starry_night.jpg")
content_img = image_loader("images/gatys/avengers_square.jpg")

# %%
unloader = transforms.ToPILImage()  # reconvert into PIL image

# plt.ion()


def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


plt.figure()
imshow(style_img, title='Style Image')

plt.figure()
imshow(content_img, title='Content Image')

# %%
# a class to calculate the content loss


class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

# %%
# fuction to calculate the Gram matrix


def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)

# %%
# a class to calculate the style loss


class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


# %%
# importing the pretrained vgg model
# pytorch's implementation of vgg has 2 sections of layers: features and classifier.
# the features section has the convolution and pooling layers and the classifier section
# has the fully connected layers.
# here we only need the features (convolution) layers.
# we are enabling the 'evaluation' mode because we dont want to train the model.
# we are not looking for the grads. We just need the feature outputs at various layers.
cnn = models.vgg19(pretrained=True).features.to(device).eval()

# %%
# vgg uses these means and stard devaition on images.
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

# create a module to normalize input image so we can easily put it in a
# nn.Sequential


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


# %%
# desired depth layers to compute style/content losses :
content_layers_default = ['conv5_1']
# style_layers_default = ['conv_1', 'conv_2', 'conv_3']
# style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4']
style_layers_default = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']


def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)

    # normalization module
    normalization = Normalization(
        normalization_mean, normalization_std).to(device)

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    l = 1
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv{}_{}'.format(l, i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
            l += 1
            i = 0
            # as suggested in the paper replacing maxpool with avgpool
            # for better results
            layer = nn.AvgPool2d(kernel_size=2)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(
                layer.__class__.__name__))

        model.add_module(name, layer)

        print(name)
        print(layer)

        if name in content_layers:
            print("layer name(content): {}".format(name))
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            print("layer name(style): {}".format(name))
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses


# %%
input_img = content_img.clone()
# if you want to use white noise instead uncomment the below line:
# input_img = torch.randn(content_img.data.size(), device=device)

# add the original input image to the figure:
plt.figure()
imshow(input_img, title='Input Image')

# %%


def get_input_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer

# %%


def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=300,
                       style_weight=1e7, content_weight=1):
    """Run the style transfer."""
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
                                                                     normalization_mean, normalization_std, style_img, content_img)
    optimizer = get_input_optimizer(input_img)

    loss_vals = []

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss_vals.append(loss.item())
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

    # a last correction...
    input_img.data.clamp_(0, 1)

    return input_img, loss_vals


# %%
loss_vals = []
output, loss_vals = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                       content_img, style_img, input_img, num_steps=1)

# %%
plt.figure()
imshow(output, title='Output Image')

# sphinx_gallery_thumbnail_number = 4
# plt.ioff()
plt.show()

# %%
utils.save_image(
    output, "images/gatys/results/avengers__starry_night_1e7_better.png")


# %%
plt.title("Total Loss vs iterations")
plt.xlabel("Iterations (every 5th)")
plt.ylabel("Loss_Total")
plt.plot(range(len(loss_vals[::10])), loss_vals[::10])
# plt.show()

plt.savefig("images/gatys/results/loss_scream_1e7_avg_every_other_5.png")

# %%
import pickle

loss_vals_file = open("images/gatys/results/loss_vals.pkl", "wb")
pickle.dump(loss_vals, loss_vals_file)
loss_vals_file.close()

# %%

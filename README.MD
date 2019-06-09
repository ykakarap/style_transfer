# Project : Style Transfer

Project description
=======================
This is a style transfer project developed by Yuvraj Balaji, Rithesh R N & Hanisha Kilari. In this project we will implement 2 variation of style transfer.  

The first implementation is a paper by Gatys et al. ([paper](https://arxiv.org/pdf/1508.06576.pdf))  

<img src="https://github.com/ykakarap/style_transfer/blob/master/sample.png" height="200" align="center">

The second implementation is cyclic GAN. ([paper](https://arxiv.org/pdf/1703.10593.pdf))

Requirements
=======================
For Gatys et al. model, the following packages are required:
- numpy
- torch
- torchvision

For cyclic GAN model, run 'requirements.txt' to install all the neceassary packages.

Code organization
=======================
#### Gatys et al. Model
```
gatys_demo/  
    demo.ipynb              -- Run to read the content and style images and generate the aritistic image  
    images/  
        avengers_square.jpg -- The content image used in demo
        starry_night.jpg    -- The style image used in demo
```
#### Cyclic GAN Model
```
cyclic-GAN/
Demonstration.ipynb     -- Run the demo of cyclic GAN for Image-to-Image translation. (Outputs the result of Real Image translation into Cezanne, Monet, Ukiyoe, Vangogh)<br>
train.py                -- To train the cyclic-GAN model.<br>
test.py                 -- To test the cyclic-GAN model.<br>
cycle_GAN_model.py      -- cycle GAN model.<br>
options                 -- training and testing options (e.g. epochs, batch_size etc)
```
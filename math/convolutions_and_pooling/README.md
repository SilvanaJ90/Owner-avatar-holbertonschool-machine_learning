# Convolutions and Pooling

## Learning Objectives

![This is an image](https://github.com/Machinelearninguru/Image-Processing-Computer-Vision/blob/master/Convolutional%20Neural%20Network/Convolutional%20Layers/_images/stride2.gif?raw=true)

At the end of this project, you are expected to be able to explain to anyone, without the help of Google:
General

    What is a convolution?
    What is max pooling? average pooling?
    What is a kernel/filter?
    What is padding?
    What is “same” padding? “valid” padding?
    What is a stride?
    What are channels?
    How to perform a convolution over an image
    How to perform max/average pooling over an image

## Requirements
General

    Allowed editors: vi, vim, emacs
    All your files will be interpreted/compiled on Ubuntu 20.04 LTS using python3 (version 3.8)
    Your files will be executed with numpy (version 1.19.2)
    All your files should end with a new line
    The first line of all your files should be exactly #!/usr/bin/env python3
    A README.md file, at the root of the folder of the project, is mandatory
    Your code should use the pycodestyle style (version 2.6)
    All your modules should have documentation (python3 -c 'print(__import__("my_module").__doc__)')
    All your classes should have documentation (python3 -c 'print(__import__("my_module").MyClass.__doc__)')
    All your functions (inside and outside a class) should have documentation (python3 -c 'print(__import__("my_module").my_function.__doc__)' and python3 -c 'print(__import__("my_module").MyClass.my_function.__doc__)')
    Unless otherwise noted, you are not allowed to import any module except import numpy as np
    You are not allowed to use np.convolve
    All your files must be executable
    The length of your files will be tested using wc

## More Info
Testing

Please download this dataset for use in some of the following main files.
https://s3.eu-west-3.amazonaws.com/hbtn.intranet.project.files/holbertonschool-ml/animals_1.npz
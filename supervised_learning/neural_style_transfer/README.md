# Neural Style Transfer



![image](https://assets-global.website-files.com/5ef788f07804fb7d78a4127a/61d6a66980d1a52a2ba28505_Neural%20Style%20Transfer.jpeg)

## Learning Objectives

At the end of this project, you are expected to be able to explain to anyone, without the help of Google:
General

    What is Neural Style Transfer?
    What is a gram matrix?
    How to calculate style cost
    How to calculate content cost
    What is Gradient Tape and how do you use it?
    How to perform Neural Style Transfer

## Requirements
General

    Allowed editors: vi, vim, emacs
    All your files will be interpreted/compiled on Ubuntu 20.04 LTS using python3 (version 3.8)
    Your files will be executed with numpy (version 1.19.2) and tensorflow (version 2.6)
    All your files should end with a new line
    The first line of all your files should be exactly #!/usr/bin/env python3
    A README.md file, at the root of the folder of the project, is mandatory
    Your code should use the pycodestyle style (version 2.6.0)
    All your modules should have documentation (python3 -c 'print(__import__("my_module").__doc__)')
    All your classes should have documentation (python3 -c 'print(__import__("my_module").MyClass.__doc__)')
    All your functions (inside and outside a class) should have documentation (python3 -c 'print(__import__("my_module").my_function.__doc__)' and python3 -c 'print(__import__("my_module").MyClass.my_function.__doc__)')
    Unless otherwise noted, you are not allowed to import any module except import numpy as np and import tensorflow as tf
    All your files must be executable

## Data

For the following main files, please use these images:

golden_gate.jpg:

![image](https://s3.eu-west-3.amazonaws.com/hbtn.intranet/uploads/medias/2019/9/d714b5d5b0e34b796e79.jpg?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA4MYA5JM5DUTZGMZG%2F20240317%2Feu-west-3%2Fs3%2Faws4_request&X-Amz-Date=20240317T235953Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=021c3727060d623333ee2abc35841b043f5cb56cee59d177a4c33398c73b9e75)


![image](https://s3.eu-west-3.amazonaws.com/hbtn.intranet/uploads/medias/2019/9/f752f0326ffe1f8af36e.jpg?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA4MYA5JM5DUTZGMZG%2F20240317%2Feu-west-3%2Fs3%2Faws4_request&X-Amz-Date=20240317T235953Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=754259df64e8b41832b07cb623a7b392bcefb5454cf71412ab237430a46dadde)
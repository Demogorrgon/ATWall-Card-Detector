# Installation

1. Install python 3.8 and 3.10 on the machine (or use pyenv to install required version)
<br />
<br />
2. Install Anaconda software on the machine
<br />
<br />
3. Walk through the official tf object detection api guide and install all needed components: https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html# 
(make sure to download pre-trained models (frcnn_resnet50_v1) for fine-tuning, and place them into ```models``` directory)
<br />
<br />
4. Set up conda env for training/testing the model: ```conda create --name <env> --file <this file>```
<br />
<br />
5. Set up python virtual env for running the api service: 
   1. ```cd api```
   2. ```python -m venv venv```
   3. ```pip install -r .\requirements.txt```
<br />
<br />

# Installation

1. Install Python 3.8 and 3.10 on the machine (or use pyenv to install the required version).
   
2. Install Anaconda software on the machine.
   
3. Follow the official tf object detection API guide to install all the needed components: [https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html)
   (Make sure to download pre-trained models (frcnn_resnet50_v1) for fine-tuning and place them into the `models` directory).
   
4. Set up a Python virtual environment for running the API service:
- Navigate to the `api` directory:
  ```
  cd api
  ```
- Create a virtual environment:
  ```
  python -m venv venv
  ```
- Install the required packages:
  ```
  pip install -r requirements.txt
  ```
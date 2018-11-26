# EM Capsnet Performance Analysis
#### Based on the implementation of https://github.com/www0wwwjs1/Matrix-Capsules-EM-Tensorflow

## Requirements
- Python >= 3.4
- Numpy
- Tensorflow >= 1.2.0
- Keras

```pip install -r requirement.txt```

## Usage
**Step 1.**
Clone this repository with ``git``.

```
$ git clone https://github.com/www0wwwjs1/Matrix-Capsules-EM-Tensorflow.git
$ cd Matrix-Capsules-EM-Tensorflow
```

**Step 2.**
Download the [MNIST dataset](http://yann.lecun.com/exdb/mnist/), ``mv`` and extract it into ``data/mnist`` directory.(Be careful the backslash appeared around the curly braces when you copy the ``wget `` command to your terminal, remove it)

```
$ mkdir -p data/mnist
$ wget -c -P data/mnist http://yann.lecun.com/exdb/mnist/{train-images-idx3-ubyte.gz,train-labels-idx1-ubyte.gz,t10k-images-idx3-ubyte.gz,t10k-labels-idx1-ubyte.gz}
$ gunzip data/mnist/*.gz
```

***To install smallNORB, follow instructions in ```./data/README.md```***

**Step 3.**
Start the training(MNIST):
```
$ python3 train.py "mnist"
```

**Step 4.**
Download the [Fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist), ``mv`` and extract it into ``data/fashion_mnist`` directory.(Be careful the backslash appeared around the curly braces when you copy the ``wget `` command to your terminal, remove it)

```
$ mkdir -p data/fashion_mnist
$ wget -c -P data/fashion_mnist http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/{train-images-idx3-ubyte.gz,train-labels-idx1-ubyte.gz,t10k-images-idx3-ubyte.gz,t10k-labels-idx1-ubyte.gz}
$ gunzip data/fashion_mnist/*.gz
```

Start the training(smallNORB):
```
$ python3 train.py "smallNORB"
```

Start the training(CNN baseline):
```
$ python3 train_baseline.py "smallNORB"
```

**Step 4.**
View the status of training:
```
$ tensorboard --logdir=./logdir/{model_name}/{dataset_name}/train_log/
```
Open the url tensorboard has shown.

**Step 5.**
Start the test on MNIST:
```
$ python3 eval.py "mnist" "caps"
```

Start the test on smallNORB:
```
$ python3 eval.py "smallNORB" "caps"
```

**Step 6.**
View the status of test:
```
$ tensorboard --logdir=./test_logdir/{model_name}/{dataset_name}/
```
Open the url tensorboard has shown.

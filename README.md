# EM Capsnet Performance Analysis
## Experiment 
Capsule network is a new proposed neural network in
[GEH18](https://openreview.net/pdf?id=HJWLfGWRb) as an enhancement to Convlutional
Neural Network(CNN) and potentially could replace CNNs in different application that require
more accurate detection of objects that tends to change in terms of graphical properties such
as position, orientation and thickness. In this work, we study the architecture of a capsule
network, demonstrate its bottlenecks by profiling its operation and finally we propose potential
solutions that can take advantage of patterns we have observed that can be implemented as a
hardware accelerator. 
### Based on the implementation of [Matrix-Capsules](https://github.com/www0wwwjs1/Matrix-Capsules-EM-Tensorflow)


## Reproduce
**Step 1.**
Clone this repository with ``git``.
```
$ git clone https://github.com/maomran/CapsNet.git
$ cd CapsNet
```
**Step 2.**
Download the [MNIST dataset](http://yann.lecun.com/exdb/mnist/)
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
Start the training(smallNORB):
```
$ python3 train.py "smallNORB"
```
Start the training(CNN baseline):
```
$ python3 train_baseline.py "smallNORB"
```

**Step 3.**
Start the test on MNIST:
```
$ python3 test.py "mnist" "caps"
```

Start the test on smallNORB:
```
$ python3 test.py "smallNORB" "caps"
```
**Step 3.**
Results are generated in folder ```results```

**Step 4.**
View your profiling on tensorboard. 

## Results
![Test Accuracy](./imgs/accuracy.png)



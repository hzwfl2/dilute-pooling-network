# `Dilute Pooling Network in keras`

## Paper 
Please cite our papers if you find it useful for your research.

## Installation
* Install `Keras 2.2.4`, `tensorflow 1.12.0`, `scipy`, `numpy` and `opencv3`.
* Or use our prepared gpu env `dpnets.yaml`
```
conda env create -f dpnets.yaml
```
* Clone this repo.
```
git clone https://github.com/snowzm/dilute-pooling
```

## Usage
* Firstly, Run `generate_samples.py` to generate the meta data.
* Then, Run `main.py`.
* Tip: DPNets model is implemented in `model.py`.


## Result
* About 100 Epoch, you will see the similar results like the following.
[==============================] - 3s 121ms/step - loss: 0.0077 - acc: 0.9979 - val_loss: 0.0187 - val_acc: 0.9963
* Other experiments script also can be found in the `main.py`.

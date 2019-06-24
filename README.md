# Keras implementation of our method `Dilute Pooling Network`

## Paper
Please cite our papers if you find it useful for your research.

```

## Installation
* Install `Keras 2.2.4`, `tensorflow 1.12.0`, `scipy`, `numpy` and `opencv3`.

* Clone this repo.
```
git clone
```

## Usage
* Firstly, Run `generate_samples.py` to generate the meta data.
* Then, Run `main.py`.
* DPNets model is implemented in `model.py`.


## Result
* About 100 Epoch, you will see the similar results like the following.
28/28 [==============================] - 3s 121ms/step - loss: 0.0077 - acc: 0.9979 - val_loss: 0.0187 - val_acc: 0.9963
* Other experiments script also can be found in the `main.py`.
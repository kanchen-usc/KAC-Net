# KAC-Net
This repository contains tensorflow implementation for *Knowledge Aided Consistency for Weakly Supervised Phrase Grounding* in [CVPR 2018](https://arxiv.org/pdf/1803.03879).

## Setup

*Note*: Please read the feature representation files in ```feature``` and ```annotation``` directories before using the code.

**Platform:** Tensorflow-1.1.0<br/>
**Visual features:** We use [Faster-RCNN](https://github.com/endernewton/tf-faster-rcnn) fine-tuned on [Flickr30K Entities](http://web.engr.illinois.edu/~bplumme2/Flickr30kEntities/). Afer fine-tuning, please put visual features in the ```feature``` directory (More details can be seen in the [```README.md```](./feature/README.md) in this directory).<br/>
**Sentence features:** We encode one-hot vector for each query, as well as the annotation for each query and image pair. Pleae put the encoded features in the ```annotation``` directory (More details are provided in the [```README.md```](./annotation/README.md) in this directory).<br/>
**File list:** We generate a file list for each image in the Flickr30K Entities. If you would like to train and test on other dataset (e.g. [Referit Game](http://tamaraberg.com/referitgame/)), please follow the similar format in the ```flickr_train_val.lst``` and ```flickr_test.lst```.<br/>
**Hyper parameters:** Please check the ```Config``` class in the ```train.py```.

## Training & Test

For training, please enter the root folder of ```KAC-Net```, then type
```
$ python train.py -m [Model Name] -g [GPU ID]
```
For testing, please entre the root folder of ```KAC-Net```, then type
```
$ python evaluate.py -m [Model Name] -g [GPU ID] --restore_id [Restore epoch ID]
```
Make sure the model name entered for evaluation is the same as the model name in training, and the epoch id exists.

## Reference

If you find the repository is useful for your research, please consider citing the following work:

```
@inproceedings{Chen_2018_CVPR,
  title={Knowledge Aided Consistency for Weakly Supervised Phrase Grounding},
  author={Chen, Kan and Gao, Jiyang and Nevatia, Ram},
  booktitle={CVPR},
  year={2018}
}
```

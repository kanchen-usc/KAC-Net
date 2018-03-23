# KAC-Net
This repository contains tensorflow implementation for *Knowledge Aided Consistency for Weakly Supervised Phrase Grounding* in [CVPR 2018](https://arxiv.org/pdf/1803.03879).

## Setup

*Note*: Please read the feature representation files in ```feature``` and ```annotation``` directories before using the code.

**Platform:** Tensorflow-1.1.0 (python 2.7)<br/>
**Visual features:** We use [Faster-RCNN](https://github.com/endernewton/tf-faster-rcnn) fine-tuned on [Flickr30K Entities](http://web.engr.illinois.edu/~bplumme2/Flickr30kEntities/). After fine-tuning, please put visual features in the ```feature``` directory (More details can be seen in the [```README.md```](./feature/README.md) in this directory).<br/>
**Global features:** We extract the global visual feature for each image in Flickr30K Entities using a pre-trained Faster-RCNN on PASCAL VOC 2012 and store them in the folder ```global_feat```.<br/>
**Sentence features:** We encode one-hot vector for each query, as well as the annotation for each query and image pair. Please put the encoded features in the ```annotation``` directory (More details are provided in the [```README.md```](./annotation/README.md) in this directory).<br/>
**File list:** We generate a file list for each image in the Flickr30K Entities. If you would like to train and test on other dataset (e.g. [Referit Game](http://tamaraberg.com/referitgame/)), please follow the similar format in the ```flickr_train_val.lst``` and ```flickr_test.lst```.<br/>
**Hyper parameters:** Please check the ```Config``` class in the ```train.py```.

## Training & Test

Before training, we first pre-train [GroundeR model](https://arxiv.org/pdf/1511.03745.pdf) (unsupervised scenario) and save the pre-trained model in the folder ```model/ground_unsupervised_base``` (epoch 53). The implementation of GroundeR is in this [repository](https://github.com/kanchen-usc/GroundeR).

For training, please enter the root folder of ```KAC-Net```, then type
```
$ python train.py -m [Model Name] -g [GPU ID] -k [knowledge]
```
You can choose different types of knowledge (```-k``` option) as KBP values: ```coco``` and ```hard_coco``` are for soft and hard KBP values with a Faster-RCNN pre-trained on MSCOCO respectively. ```pas``` and ```hard_pas``` are for soft and hard KBP values with a VGG Network pre-trained on PASCAL VOC 2012 respectively. More details can be found in the [paper](https://arxiv.org/pdf/1803.03879).

For testing, please enter the root folder of ```KAC-Net```, then type
```
$ python evaluate.py -m [Model Name] -g [GPU ID] -k [knowledge] --restore_id [Restore epoch ID]
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

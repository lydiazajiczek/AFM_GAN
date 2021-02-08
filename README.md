# AFM GAN
Style transfer architecture for inferring pixelwise elastic modulus values of human tissue samples from grayscale microscope images. Based on the Pix2Pix architecture implementation in Keras by [Erik Linder-Norén](https://github.com/eriklindernoren/Keras-GAN/tree/master/pix2pix).

## Functions
* `train.py`: takes the following command line arguments:
	* `--dataset_name` - the dataset to train on (default: `'liver'`)
	* `--epochs` - integer number of training epochs (default 10000)
	* `--batch_size` - integer batch size (default 16)
	* `--model_name` - (required) the .h5 weights file to save to (**without the file extension**)

* `predict.py`: takes the following command line arguments:
	 * `--fn` - (required) the filename of the unstained TIFF image to predict on (**without the file extension**)
	* `--nr` and `--nc` - the integer number of subrows and subcolumns to split the image into to save memory (default 8)
	* `--model_name` - (required) the saved .h5 weights file to use for prediction (**also without the file extension**)
* `data_loader.py`: `DataLoader` class is used during training and testing for loading batches of images, preprocessing them and performing data augmentation as well as processing subimages during prediction.
* `GAN.py`: instantiates the GAN class. Edit this to change the network architecture.

## Installation and Testing
Tested on Windows 10 and Ubuntu 18.04.3 LTS.

1. Install Python 3.5.2/3.68 with the following packages:
	* tensorflow-gpu (tested on 1.11.0/1.12.0 with CUDA 9.0)
	* keras (tested with 2.24)
	* h5py
	* Pillow
	* numpy
	* opencv-python
2. Clone the github repo
3. Download the training dataset [here](https://weiss-develop.cs.ucl.ac.uk/afm-data/afm-dataset.zip) and extract into a folder called `data`
4. Run
	* `train.py --dataset_name liver --epochs 10000 --batch_size 16 --model_name liver`
5. Once it has finished training, run
	* `predict.py --fn test --nr 8 --nc 8 --model_name liver`

Typical times to predict on`test.tiff` (3672 x 3282 11.4 MB image):
* 14 minutes on NVIDIA Tesla V100-DGXS 32GB (Ubuntu 18.04)
* 18 minutes on NVIDIA GeForce GTX 1050 Ti 4GB (Windows 10)

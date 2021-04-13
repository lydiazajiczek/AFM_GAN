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

To fully test, follow all the below steps. To just test prediction using a pre-trained model (found in the `testing` folder), skip steps 3 and 4.

1. Install Python 3.5.2/3.68 with the following packages:
	* tensorflow-gpu (tested with version 1.11.0/1.12.0 using CUDA 9.0)
	* keras (tested with version 2.24)
	* h5py
	* Pillow
	* numpy
	* opencv-python
2. Clone the github repo
3. Download the training dataset [here](https://weiss-develop.cs.ucl.ac.uk/afm-liver-tissue-data/training_data.zip) and extract into the cloned folder
4. Run
	* `train.py --dataset_name liver --epochs 10000 --batch_size 16 --model_name liver`
5. Once it has finished training (unless training was skipped), run
	* `predict.py --fn test --nr 8 --nc 8 --model_name liver`

The output will be a file named `test_pred.tiff`, however it is a 32-bit float image so will not display properly in most photo editors. `test_pred.png` is provided in the `testing` folder as a prediction TIFF that been loaded in ImageJ, converted to a pseudocolor image using a lookup table and adjusted so that only values between 0 and 2 kPa are displayed. The image mask used to remove pixels corresponding to bubbles, tissue glue, background or out of focus regions is provided in `test_mask.tiff` which is also provided as a human-readable PNG file. The final masked prediction test image is provided as `test_pred_masked.tiff` and `test_pred_masked.png`.

Typical times to train on the liver dataset:
* 3 hours on NVIDIA Tesla V100-DGXS 32GB (Ubuntu 18.04)
* 5 hours on NVIDIA GeForce GTX 1050 Ti 4GB (Windows 10)
* 19 hours on CPU (Intel i9-8950HK CPU, Windows 10)

Typical times to predict on `test.tiff` (3672 x 3282 11.4 MB image):
* 14 minutes on NVIDIA Tesla V100-DGXS 32GB (Ubuntu 18.04)
* 18 minutes on NVIDIA GeForce GTX 1050 Ti 4GB (Windows 10)
* 41 minutes on CPU (Intel i9-8950HK CPU, Windows 10)

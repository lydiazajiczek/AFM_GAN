from GAN import GAN
import argparse


def arg_parser():
	# Defines an argument parser for the script
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('--fn', type=str, required=True,
						help='Filename of unstained image TIFF to predict, without file extension')
	parser.add_argument('--nr', type=int, default=8,
						help='Number of subimage rows (default 8)')
	parser.add_argument('--nc', type=int, default=8,
						help='Number of subimage columns (default 8)')
	parser.add_argument('--model_name', type=str, required=True,
						help='Name of .h5 model file, without file extension')
	return parser


if __name__ == '__main__':
	args = arg_parser().parse_args()
	gan = GAN(input_type="fresh", output_type="stiffness", model_name=args.model_name)
	gan.predict(args.fn, args.nr, args.nc)

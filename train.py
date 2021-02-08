from GAN import GAN
import argparse


def arg_parser():
    # Defines an argument parser for the script
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset_name', type=str, default="liver",
                        help="Name of training dataset (default: liver)")
    parser.add_argument('--epochs', type=int, default=10000,
                        help="Number of training epochs (default: 10000)")
    parser.add_argument('--batch_size', type=int, default=16,
                        help="Batch size (default: 16)")
    parser.add_argument('--model_name', type=str, required=True,
                        help="Name of .h5 weights file (without file extension)")
    return parser


if __name__ == '__main__':
    args = arg_parser().parse_args()
    gan = GAN(dataset_name=args.dataset_name, input_type="fresh", output_type="stiffness", model_name=args.model_name)
    gan.train(epochs=args.epochs, batch_size=args.batch_size)

from argparse import ArgumentParser
from webmnist.export import export
from webmnist.train import train


parser = ArgumentParser()
parser.add_argument("-o", "--output", type=str, required=True)
parser.add_argument("-e", "--epochs", type=int, default=3)
parser.add_argument(      "--train",  action="store_true")
parser.add_argument(      "--export", action="store_true")
args = parser.parse_args()

if args.train: train(args.output, epochs=args.epochs)
if args.export: export(args.output)
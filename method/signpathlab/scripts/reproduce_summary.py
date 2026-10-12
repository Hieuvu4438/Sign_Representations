import argparse
from signpath.experiments import summarize

parser = argparse.ArgumentParser()
parser.add_argument("--root", default="artifacts/verified_pilot")
parser.add_argument("--output", default="artifacts/verified_pilot/comparison.csv")
args = parser.parse_args()
print(summarize(args.root, args.output))

import _bootstrap
import argparse
import yaml
from signrel.engine.train import train
from signrel.config import load_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config",required=True)
    p.add_argument("--resume")
    p.add_argument("--stop-after",type=int,help="Stop at this global step without altering LR/EMA schedule")
    p.add_argument("--run-dir")
    p.add_argument("--device",choices=["cuda","cpu"])
    p.add_argument("--seed",type=int)
    args = p.parse_args()
    config = load_config(args.config)
    if args.run_dir:
        config["trainer"]["run_dir"] = args.run_dir
    if args.device:
        config["trainer"]["device"] = args.device
    if args.seed is not None:
        config["seed"] = args.seed
    train(config,args.resume,args.stop_after)


if __name__ == "__main__":
    main()

import argparse
import pathlib

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", type = pathlib.Path, 
        help = "path to folder containing events")
    args = parser.parse_args()
    event_files = list(args.log_dir.glob("events.out.tfevents.*"))
    if not event_files or len(event_files) > 1:
        raise FileNotFoundError(f"only expect one event file in {args.log_dir}")
    event_file = event_files[0]
    ea = EventAccumulator(str(event_file))
    ea.Reload()
    epochs = ea.Scalars("epoch")
    if epochs:
        print(f"last epoch:  {epochs[-1].value}")
    miou = ea.Scalars("val/Mean_IOU")
    if miou:
        print(f"best mIoU: {max(miou, key = lambda event: event.value).value:.6f}")

if __name__ == "__main__":
    main()

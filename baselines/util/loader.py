
import pathlib
import argparse

def resolve(base_path: pathlib.Path, extension):
    files = list(base_path.rglob(f"*{extension}"))
    if not files: return None
    if len(files) > 1:
        raise RuntimeError(f'expected a single {extension}, but found multiple: {base_path}')
    return files[0].resolve(), base_path.name

class Loader:
    def parser():
        p = argparse.ArgumentParser()
        p.add_argument("path_to_bags")
        p.add_argument("path_to_result")
        p.add_argument("--path_to_config")
        p.add_argument("--name")
        return p
        
    def yield_config(self):
        stem = ""
        if self.path_to_config is not None:
            if self.path_to_config.is_dir():
                stem = self.path_to_config.stem if self.name is None else self.name
            if len(stem) > 0:
                stem += '_'

        for config in self.configs:
            for bag, metadata in self.bags:
                bag_path, bag_name = bag
                metadata_path, _ = metadata

                name = self.name
                if config is not None:
                    if self.path_to_config.is_dir() or self.name is None:
                        name = config.stem
                    name = stem + name

                result = (self.path_to_result / name / 'digiforests' / bag_name).resolve()
                result.mkdir(parents = True, exist_ok = True)

                sma_config = (self.path_to_result / name / 'config.yaml').resolve()
                sma_config.write_text('category: odometry')

                yield name, config, bag_path, metadata_path, result

    def __init__(self, args):
        self.path_to_bags = pathlib.Path(args.path_to_bags)
        if not self.path_to_bags.exists():
            raise RuntimeError(f'path_to_bags does not exist: {args.path_to_bags}')
        self.path_to_result = pathlib.Path(args.path_to_result)
        if not self.path_to_result.exists():
            self.path_to_result.mkdir(exist_ok = True, parents = True)
        self.path_to_config = pathlib.Path(args.path_to_config) if args.path_to_config else None

        self.bags = [(resolve(p, extension = '.mcap'), resolve(p, extension = '.yaml')) for p in self.path_to_bags.iterdir() if p.is_dir() and p not in self.path_to_result.parents]
        self.bags = list(filter(lambda p: p[0] is not None and p[1] is not None, self.bags))

        self.name = args.name

        if self.path_to_config is not None:
            if self.path_to_config.is_dir():
                self.configs = [f.resolve() for f in self.path_to_config.iterdir() if f.is_file()]
            else:
                self.configs = [self.path_to_config.resolve()]
        elif self.name is None:
            raise RuntimeError('must supply name if path_to_config is missing')
        else:
            self.configs = [None]

        self.generator = self.yield_config()

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.generator)
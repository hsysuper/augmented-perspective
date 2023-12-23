import abc


class BaseDepthModel:

    def __init__(self, name: str, args, parser):
        self.name = name
        if args.image_path.is_dir():
            vars(args)['image_path'] = '../../' + str(args.image_path)
            vars(args)['image_files'] = None
        else:
            path = args.image_path
            vars(args)['image_path'] = '../../' + str(path.parent)
            vars(args)['image_files'] = [str(path.name)]
        vars(args)['output_path'] = '../../' + str(args.output_path)
        self.args = args
        self.parser = parser

    @abc.abstractmethod
    def get_depth_map(self):
        pass

    @staticmethod
    def require_normalization():
        return False

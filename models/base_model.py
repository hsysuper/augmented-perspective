import abc


class BaseDepthModel:

    def __init__(self, name: str, args, parser):
        """
        Initialize the depth model class

        :param name: name of the model for logging and filename purposes
        :param args: command line arguments passed to this model
        :param parser: argument parser from upper layer
        """
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
        """
        Get the depth map calculated by this model
        :return: array of depth map in the same dimension of the input image
        """
        pass

    @staticmethod
    def require_normalization():
        """
        Whether a normalization step with scaling factor is needed for the depth map when used on this model
        :return: True or False
        """
        return False

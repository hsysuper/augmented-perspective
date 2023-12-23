import torch
import torch.nn as nn


class BaseModel(torch.nn.Module):

    def load(self, path):
        """Load model from file.

        Args:
            path (str): file path
        """
        # NOTE: Change this to cuda devices if using CUDA.
        parameters = torch.load(path, map_location='cpu')

        if "optimizer" in parameters:
            parameters = parameters["model"]

        self.load_state_dict(parameters)

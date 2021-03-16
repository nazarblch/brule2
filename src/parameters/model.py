from argparse import ArgumentParser


class ModelParameters(ArgumentParser):

    def __init__(self, *args, **kwargs):
        super().__init__(add_help=False, *args, **kwargs)
        self.add_argument('--weights', type=int, default=0)
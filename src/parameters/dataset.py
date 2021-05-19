from argparse import ArgumentParser


class DatasetParameters(ArgumentParser):

    def __init__(self, *args, **kwargs):
        super().__init__(add_help=False, *args, **kwargs)
        self.add_argument('--image_size', type=int, default=256)
        self.add_argument('--batch_size', type=int, default=8)
        self.add_argument('--measure_size', type=int, default=68)
        self.add_argument('--data_path', type=str, default="")
        self.add_argument('--data_size', type=int, default=100)





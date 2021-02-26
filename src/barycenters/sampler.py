import ot
import numpy as np
from typing import List


class Uniform2DBarycenterSampler:

    def __init__(self, size: int, dir_alpha: float = 1):

        self.X = np.random.uniform(size=(size, 2))
        self.prob = np.ones(size) / size
        self.dir_alpha = dir_alpha

    def sample(self, measures: List[np.ndarray]) -> (np.ndarray, np.ndarray):

        weights = np.random.dirichlet([self.dir_alpha] * len(measures))

        return ot.lp.free_support_barycenter(
            measures,
            [self.prob] * len(measures),
            self.X,
            self.prob,
            weights=weights
        ), weights

    def mean(self, measures: List[np.ndarray]) -> np.ndarray:

        return ot.lp.free_support_barycenter(
            measures,
            [self.prob] * len(measures),
            self.X,
            self.prob
        )
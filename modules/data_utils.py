from typing import Iterator, Sized

from torch.utils.data.sampler import Sampler, SequentialSampler, RandomSampler

import nnlib.nnlib.data_utils.base
from nnlib.nnlib.callbacks import Callback


def load_data_from_arguments(args, build_loaders=True):
    data_selector = nnlib.nnlib.data_utils.base.DataSelector()
    return data_selector.parse(args, build_loaders=build_loaders)


class SwitchableRandomSampler(Sampler):
    def __init__(self, data_source: Sized, replacement: bool = False, shuffle: bool = False) -> None:
        super(SwitchableRandomSampler, self).__init__(data_source=data_source)
        self.data_source = data_source
        self.shuffle = shuffle
        self.sequential_sampler = SequentialSampler(data_source=data_source)
        self.random_sampler = RandomSampler(data_source=data_source, replacement=replacement)

    def set_shuffle(self, value):
        self.shuffle = value

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            return self.random_sampler.__iter__()
        return self.sequential_sampler.__iter__()

    def __len__(self) -> int:
        return len(self.data_source)


class TurnOnTrainShufflingCallback(Callback):
    def __init__(self, switchable_random_sampler: SwitchableRandomSampler):
        super(TurnOnTrainShufflingCallback, self).__init__()
        self.switchable_random_sampler = switchable_random_sampler

    def call(self, epoch, *args, **kwargs):
        if epoch == 0:
            print("Turning on the random shuffling of training data")
            self.switchable_random_sampler.set_shuffle(True)

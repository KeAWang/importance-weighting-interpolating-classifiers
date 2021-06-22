from collections import namedtuple, defaultdict
from torch.utils.data import Dataset
from torch import Generator
import torch
import numpy as np
from typing import Sequence, Optional, Dict, Union


# UGH: Pytorch dataloaders don't support dataclasses, only namedtuples
# See https://github.com/pytorch/pytorch/blob/9baf75c86edc7f6cd1c04bf9f42d18bc0d05f504/torch/utils/data/_utils/collate.py#L43
LabeledDatapoint = namedtuple(
    "LabeledDatapoint", ("x", "y", "w"), defaults=(None, None, 1)
)


GroupedLabeledDatapoint = namedtuple(
    "GroupedLabeledDatapoint", ("x", "y", "w", "g"), defaults=(None, None, 1, None)
)

ExtraLabeledDatapoint = namedtuple(
    "ExtraLabeledDatapoint",
    ("x", "y", "w", "g", "extra"),
    defaults=(None, None, 1, None, False),
)


class UndersampledByGroupDataset(Dataset):
    """ Wraps a map Dataset into an dataset undersampled by group.

    Args:
        dataset (Dataset): The whole Dataset
        group_ids (Sequent[int]): Each entry i is the group id of dataset[i]
        new_group_sizes (dict or list): new_group_sizes[g] returns the desired group
        size to undersample to for group g.
        generator (Optional[Generator]): torch.Generator

    Note that we can undersample by labels if `group_ids[i]` is the label of dataset[i]
    """

    def __init__(
        self,
        dataset: Dataset,
        group_ids: Sequence[int],
        new_group_sizes: Optional[Union[Sequence[int], Dict[int, int]]] = None,
        new_group_fracs: Optional[Union[Sequence[float], Dict[int, float]]] = None,
        generator=None,
    ):
        if (new_group_sizes is not None) and (new_group_fracs is not None):
            raise ValueError(
                f"new_group_sizes and new_group_fracs cannot both be specified"
            )
        if isinstance(new_group_sizes, list):
            new_group_sizes = dict(enumerate(new_group_sizes))
        elif isinstance(new_group_fracs, list):
            new_group_fracs = dict(enumerate(new_group_fracs))

        group_idxs = defaultdict(list)
        for i, g in enumerate(group_ids):
            group_idxs[int(g)].append(i)

        if new_group_fracs is not None:
            raise NotImplementedError
            # new_group_sizes = {
            #    g: int(len(group_idxs[g]) * frac) for g, frac in new_group_fracs
            # }

        for g in group_idxs.keys():
            assert new_group_sizes[g] <= len(
                group_idxs[g]
            ), f"Group {g} has only {len(group_idxs[g])} samples, which is less than {new_group_sizes[g]} "

        indices = []
        if generator is None:
            generator = torch.Generator()
            generator.manual_seed(
                0
            )  # ensure that we always choose the same examples with a group
        for g, idxs in group_idxs.items():
            idxs = torch.tensor(idxs)
            new_size = new_group_sizes[g]
            # equivalent of np.random.choice without replacement
            choice = torch.randperm(len(idxs), generator=generator)[:new_size]
            chosen_idxs = idxs[choice]
            indices.append(chosen_idxs)

        self.indices = torch.cat(indices)
        self.dataset = dataset

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


class UndersampledDataset(Dataset):
    """ Wraps a map Dataset into an undersampled dataset.
    Points may be dropped from implementing undersampling via rejection sampling.

    tldr: This will exclude points from the given Dataset!

    Args:
        dataset (Dataset): The whole Dataset
        weights (sequence):  The importance weights of each element of `dataset`.
            weights[i] should be equal to the likelihood ratio of dataset[i] between
            the target distribution and the source distribution
        weights_upper_bound (Optional[float]): an optional upper bound on `weights`for rejection sampling
        generator (Optional[Generator]): torch.Generator
    """

    def __init__(
        self,
        dataset: Dataset,
        weights: Sequence[float],
        weights_upper_bound: Optional[float] = None,
        generator: Optional[Generator] = None,
    ):
        self.dataset = dataset
        self.weights = weights
        self.generator = generator

        assert len(weights) == len(dataset)
        weights = torch.tensor(weights)
        assert weights.ndim == 1
        if weights_upper_bound is None:
            weights_upper_bound = weights.max().item()

        unif_rv = torch.rand(weights.size(), generator=generator)
        # only keep these indices
        indices = torch.nonzero(
            unif_rv <= weights / weights_upper_bound, as_tuple=True
        )[0]
        self.indices = indices
        print("DEBUG INFO: selected ids are "+str(self.indices))

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


class OversampledDataset(Dataset):
    """Wraps a map Dataset into an oversampled dataset that oversamples by duplicating
    points in the original Dataset. All points in the original Dataset are kept

    tldr: This will include all points from the given Dataset and some duplicates!

    Args:
        dataset (Dataset): The whole Dataset
        weights (sequence):  The importance weights of each element of `dataset`.
            weights[i] should be equal to the likelihood ratio of dataset[i] between
            the target distribution and the source distribution
        new_size (int): The desired size of the new dataset
        generator (Optional[Generator]): torch.Generator
    """

    def __init__(
        self,
        dataset: Dataset,
        weights: Sequence[float],
        new_size: int,
        generator: Optional[Generator] = None,
    ):
        self.dataset = dataset
        self.weights = weights
        self.generator = generator

        assert weights.ndim == 1
        assert len(weights) == len(dataset)
        num_needed = new_size - len(dataset)
        assert num_needed >= 0

        weights = torch.tensor(weights)
        normalized_weights = weights / weights.sum(0)

        additional_indices = torch.multinomial(
            normalized_weights, num_needed, replacement=True, generator=generator
        )
        # append on additional samples
        self.indices = torch.cat([torch.arange(len(dataset)), additional_indices])

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


# class OversampledUndersampledDataset(Dataset):
#    """
#
#    """
#
#    def __init__(
#        self,
#        dataset: Dataset,
#        weights: Sequence[int],
#        exponent: float,
#        new_size: Optional[int] = None,
#        weights_upper_bound: Optional[float] = None,
#        generator: Optional[Generator] = None,
#    ):
#        # when exponent == 0, we only undersample
#        # when exponent == 1, we only oversample
#        assert 0 <= exponent <= 1
#
#        undersampling_weights = torch.tensor(weights) ** (1 - exponent)
#        undersampled_dataset = UndersampledDataset(
#            dataset,
#            weights=undersampling_weights,
#            weights_upper_bound=weights_upper_bound,
#            generator=generator,
#        )
#
#        if exponent == 0:
#            print(
#                "OversampledUndersampledDataset exponent is 0. Cannot oversample to requested new_size of {new_size}"
#            )
#            new_size = len(undersampled_dataset)
#        elif new_size is None:
#            raise ValueError("new_size cannot be None when exponent is nonzero!")
#
#        oversampling_weights = (
#            torch.tensor([weights[i] for i in undersampled_dataset.indices]) ** exponent
#        )
#        oversampled_dataset = OversampledDataset(
#            undersampled_dataset,
#            weights=oversampling_weights,
#            new_size=new_size,
#            generator=generator,
#        )
#
#        self.dataset = oversampled_dataset
#
#    def __getitem__(self, idx):
#        return self.dataset[idx]
#
#    def __len__(self):
#        return len(self.dataset)


class ReweightedDataset(Dataset):

    """Wraps a map Dataset into a reweighted dataset.

    Each time we __getitem__ we'll also get a weight `w` as part of our returned
    tuple

    Args:
        dataset (Dataset): The whole Dataset
        weights (sequence):  The importance weights of each element of `dataset`.
            weights[i] should be equal to the likelihood ratio of dataset[i] between
            the target distribution and the source distribution
        generator (Optional[Generator]): torch.Generator
    """

    def __init__(
        self,
        dataset: Dataset,
        weights: Sequence[float],
        generator: Optional[Generator] = None,
    ):

        self.dataset = dataset
        self.weights = weights
        self.generator = generator

    def __getitem__(self, idx):
        datapoint = self.dataset[idx]
        w = self.weights[idx]
        if isinstance(datapoint, tuple) and hasattr(
            datapoint, "_fields"
        ):  # check if namedtuple
            datapoint = datapoint._replace(w=w)
        else:
            datapoint = datapoint + (w,)
        return datapoint

    def __len__(self):
        return len(self.dataset)


class ResampledDataset(Dataset):
    """Wraps a map Dataset into a resampled dataset with replacement.

    tldr: This may drop some points and create some duplicated points from the given Dataset!

    Args:
        dataset (Dataset): The whole Dataset
        weights (sequence):  The importance weights of each element of `dataset`.
            weights[i] should be equal to the likelihood ratio of dataset[i] between
            the target distribution and the source distribution
        generator (Optional[Generator]): torch.Generator
    """

    def __init__(
        self,
        dataset: Dataset,
        weights: Sequence[float],
        new_size: int,
        generator: Optional[Generator] = None,
    ):
        self.dataset = dataset
        self.weights = weights
        self.generator = generator

        assert weights.ndim == 1
        assert len(weights) == len(dataset)

        weights = torch.tensor(weights)
        normalized_weights = weights / weights.sum(0)

        indices = torch.multinomial(
            normalized_weights, new_size, replacement=True, generator=generator
        )
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


def undersampling_schedule(
    weights: torch.Tensor, T: int, annealing_fn: str, annealing_params: tuple,
):

    if annealing_fn == "linear":

        def g(weights, t, T):
            return weights ** (1 - t / T)

    elif annealing_fn == "sigmoid":

        def sigmoid(x):
            return np.where(
                x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x))
            ).item()

        c = annealing_params[0]

        def g(weights, t, T):
            x = t / T
            return weights ** sigmoid(-c * (x - 0.5))

    elif annealing_fn == "step":

        def g(weights, t, T):
            return torch.ones_like(weights) if t > T // 2 else weights

    else:
        raise ValueError(f"Annealing function {annealing_fn} not supported")

    weights = weights / torch.max(weights)
    rv = torch.rand(*weights.shape)
    for t in range(T + 1):

        keep_idx = rv <= g(weights, t, T)
        idx = torch.nonzero(keep_idx, as_tuple=True)[0]

        weights_t = weights[idx] / g(weights[keep_idx], t, T)

        yield idx, weights_t


class UndersampledWithExtrasDataset(UndersampledDataset):
    def __init__(
        self,
        dataset: Dataset,
        weights: Sequence[float],
        weights_upper_bound: Optional[float] = None,
        generator: Optional[Generator] = None,
    ):
        super().__init__(
            dataset=dataset,
            weights=weights,
            weights_upper_bound=weights_upper_bound,
            generator=generator,
        )
        self.undersampled_indices = self.indices
        del self.indices
        all_indices = torch.arange(len(self.dataset))
        extra = torch.ones(len(all_indices), dtype=torch.bool)
        extra[self.undersampled_indices] = False
        self.extra = extra

    def __getitem__(self, idx):
        item = self.dataset[idx]
        extra = self.extra[idx]
        wrapped_item = item + (extra,)
        wrapped_item = ExtraLabeledDatapoint(*wrapped_item)
        return wrapped_item

    def __len__(self):
        return len(self.dataset)

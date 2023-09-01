"""
@title

@description

"""
import copy
import uuid
from pathlib import Path

import numpy as np
import torch
from numpy.random import default_rng
from torch import nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from island_influence import project_properties


def linear_stack(n_inputs, n_hidden, n_outputs):
    hidden_size = int((n_inputs + n_outputs) / 2)
    network = nn.Sequential(
        nn.Linear(n_inputs, n_outputs)
    )
    for idx in range(n_hidden):
        network.append(nn.Linear(hidden_size, hidden_size))
    network.append(nn.Linear(hidden_size, n_outputs))
    return network


def linear_layer(n_inputs, n_hidden, n_outputs):
    network = nn.Sequential(
        nn.Linear(n_inputs, n_outputs)
    )
    return network


def linear_relu_stack(n_inputs, n_hidden, n_outputs):
    hidden_size = int((n_inputs + n_outputs) / 2)
    network = nn.Sequential(
        nn.Linear(n_inputs, hidden_size),
        nn.ReLU()
    )

    for idx in range(n_hidden):
        network.append(nn.Linear(hidden_size, hidden_size))
        network.append(nn.ReLU())

    output_layer = nn.Linear(hidden_size, n_outputs)
    network.append(output_layer)
    return network


def load_pytorch_model(model_path):
    model = torch.load(model_path)
    model.eval()
    return model


class NeuralNetwork(nn.Module):

    @property
    def name(self):
        return f'{str(self.network_id)}_NN_{self.network_func.__name__}'

    @property
    def fitness(self):
        return self._fitness

    @fitness.setter
    def fitness(self, value):
        if self.learner:
            self._fitness = value
        else:
            raise RuntimeError(f'Can\'t set a fitness value on a non-learning network')
        return

    def __init__(self, n_inputs, n_outputs, n_hidden=2, learner=True, agent_type=None, network_func=linear_relu_stack):
        super(NeuralNetwork, self).__init__()
        # todo  make network funcs defines layers from ModuleDict or ParameterDict
        # https://pytorch.org/docs/stable/generated/torch.nn.ModuleDict.html#torch.nn.ModuleDict
        # https://pytorch.org/docs/stable/generated/torch.nn.ParameterDict.html#torch.nn.ParameterDict
        self.network_id = uuid.uuid1().int

        self.network_func = network_func
        self.learner = learner
        # todo  calculate fitness is a manner similar to Truefitness
        #       https://docs.battlesnake.com/guides/tournaments/hosting
        # todo  represent fitness as (mu,sigma)
        #       sample from distribution when fitness is collapsed
        #   this also implies that the same network will (almost certainly) have a
        #   different fitness value on two consecutive queries
        self._fitness_distribution = (0, 0)
        self._fitness_vector = [(0, 0)]
        self._fitness = None if learner else 0.0

        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs

        # https://pytorch.org/docs/stable/generated/torch.flatten.html
        # self.flatten = nn.Flatten()
        self.network = self.network_func(n_inputs=n_inputs, n_hidden=n_hidden, n_outputs=n_outputs)

        self.parent = None

        self.agent_type = agent_type
        return

    def __repr__(self):
        name_parts = self.name.split('_')
        base_repr = f'{name_parts[0][-8:]}_{"_".join(name_parts[1:])}'
        if hasattr(self, 'fitness'):
            base_repr = f'{base_repr}, {self.fitness=}'
        return base_repr

    def _deepcopy(self):
        # https://discuss.pytorch.org/t/deep-copying-pytorch-modules/13514/2
        new_model = copy.deepcopy(self)
        new_model.network_id = uuid.uuid1().int
        new_model.fitness = None
        # new_copy.parent = self.name
        new_model.parent = self
        return new_model
    
    def copy(self):
        """
        Creating a new object and then manually assigning the weights and desired attributes is almost twice as fast as deepcopying
        the pytorch neural network object.
        :return:
        """
        new_model = NeuralNetwork(self.n_inputs, self.n_outputs, n_hidden=0,  learner=self.learner, network_func=self.network_func)
        with torch.no_grad():
            param_vector = parameters_to_vector(self.parameters())
            vector_to_parameters(param_vector, new_model.parameters())
        new_model.fitness = None
        new_model.parent = self
        return new_model

    def mutate_gaussian_individual(self, mutation_scalar=0.1, probability_to_mutate=0.05):
        """
        This is slowed down by looping over each parameter value and generating a value to check if it is altered.
        There also appears to be a very small value error with fewer values than specified being selected.

        :param mutation_scalar:
        :param probability_to_mutate:
        :return:
        """
        rng = default_rng()
        with torch.no_grad():
            param_vector = parameters_to_vector(self.parameters())

            num_altered = 0
            for each_val in param_vector:
                rand_val = rng.random()
                if rand_val <= probability_to_mutate:
                    noise = torch.randn(each_val.size()) * mutation_scalar
                    each_val.add_(noise)
                    num_altered += 1

            vector_to_parameters(param_vector, self.parameters())
        num_vals = param_vector.size(dim=0)
        return num_altered, num_vals

    def mutate_gaussian(self, mutation_scalar=0.1, probability_to_mutate=0.05):
        with torch.no_grad():
            param_vector = parameters_to_vector(self.parameters())
            noise_vector = torch.randn_like(param_vector)
            noise_vector *= mutation_scalar

            ignore_vector = torch.rand_like(param_vector)
            ignore_vector -= probability_to_mutate

            # noinspection PyTypeChecker
            ignore_vector = torch.where(ignore_vector < 0, 1, 0)
            num_altered = torch.sum(ignore_vector)
            noise_vector = noise_vector.masked_fill(ignore_vector <= 0, 0)
            param_vector += noise_vector

            vector_to_parameters(param_vector, self.parameters())
        num_vals = param_vector.size(dim=0)
        return num_altered.numpy(), num_vals

    def replace_layers(self, other_net, layers):
        # todo  allow swap specific weights in each_layer
        #       this could allow for a crossover mutation where
        #       an agent considers a state input the way it considered
        #       another state input
        with torch.no_grad():
            other_children = list(other_net.network.children())
            for idx, child in enumerate(self.network.children()):
                if idx in layers:
                    other_weights = copy.deepcopy(other_children[idx].weight)
                    child.weight = other_weights
        return

    def weight_vectors(self):
        with torch.no_grad():
            weights = [
                child.weight.numpy() if hasattr(child, 'weight') else np.asarray([])
                for child in self.network.children()
            ]
        return weights

    def device(self):
        dev = next(self.parameters()).device
        return dev

    def forward(self, x):
        # todo  optimize pytorch forward pass
        # https://discuss.pytorch.org/t/updating-multiple-networks-simultaneously-on-cpu/94742/4
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)

        if x.dtype is not torch.float32:
            x = x.float()

        # if x.shape[0] != self.n_inputs:
        #     # if input does not have the correct shape
        #     # x = torch.zeros([1, self.n_inputs], dtype=torch.float32)
        #     raise ValueError(f'Input does not have correct shape: {x.shape=} | {self.n_inputs=}')

        logits = self.network(x)
        return logits

    def forward1(self, x):
        """
        This is hardly faster, despite having no checks. When using the built-in pytorch __call__ functionality, it actually appears to be slower.
        :param x:
        :return:
        """
        x = torch.from_numpy(x)
        if x.dtype is not torch.float32:
            x = x.float()
        logits = self.network(x)
        return logits

    def save_model(self, save_dir=None, tag=''):
        # todo optimize saving pytorch model
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training
        if save_dir is None:
            save_dir = project_properties.model_dir

        if not save_dir.exists():
            save_dir.mkdir(parents=True, exist_ok=True)

        if tag != '':
            tag = f'_{tag}'

        save_name = Path(save_dir, f'{self.name}_model{tag}.pt')
        torch.save(self, save_name)
        return save_name

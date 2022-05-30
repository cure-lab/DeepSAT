from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .circuit_dataset import CircuitDataset
from .neurosat_dataset import NeuroSATDataset
from .circuitsat_dataset import CircuitSATDataset

dataset_factory = {
  'deepsat': CircuitDataset,
  'cnf': NeuroSATDataset,
  'circuitsat': CircuitSATDataset
}
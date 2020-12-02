#! /usr/bin/python3
import numpy as np
import tensorflow as tf
import sys
from build_dataset import build_dataset,get_num_elements_in_dataset


ds = build_dataset(sys.argv[1:], 5000)

print(get_num_elements_in_dataset(ds))

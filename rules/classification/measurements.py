from scipy.stats import entropy
import numpy as np


def calculate_entropy(y, classes_count):
    counts = np.unique(y, return_counts=True)[1]
    return entropy(counts, base=classes_count)

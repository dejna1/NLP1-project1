import itertools

import numpy as np
import pandas as pd

from tut_rope import BenavoliCorrelatedT


def get_measurements(metrics, representation, architecture, metric="accuracy"):
    """
    Extracts the cross-validation measurements for one of the 9 possible
    combinations. Compares based on accuracy.
    """
    # selects row that matches the requested rpresentation and requested
    # architecture by checking the metrics(.csv)
    selected = metrics[(metrics["representation"] == representation)
                       & (metrics["architecture"] == architecture)]

    return selected[metric].to_numpy()


def compare_pair(metrics, representation_a, architecture_a, representation_b,
                 architecture_b, metric="accuracy", rope_size=0.005):
    """
    Compares a specific pair of representation x model combination, based
    on accuracy, using the default rope size of 0.005 (from tutorial).
    
    It reports the posterior probability whether the system is better or worse,
    by calculating the posterior model to their differences.
    """
    measurements_a = get_measurements(metrics, representation_a,
                                      architecture_a, metric)

    measurements_b = get_measurements(metrics, representation_b, architecture_b,
                                      metric)

    # systems are evaluated on the same train/validation partitions, so their
    # fold-level errors are not independent - correlated assumption
    posterior = BenavoliCorrelatedT(measurements_a, measurements_b,
                                    names=[f"{representation_a}__{architecture_a}",
                                           f"{representation_b}__{architecture_b}"])

    # probability_a = the probability that a is better
    probability_a, probability_tie, probability_b = posterior.probs(rope_size)

    return {
        "system_a": f"{representation_a}__{architecture_a}",
        "system_b": f"{representation_b}__{architecture_b}",
        "metric": metric,
        "rope_size": rope_size,
        # average value of ACCURACY (chosen metric) across seeds and folds
        "mean_a": measurements_a.mean(),
        "mean_b": measurements_b.mean(),
        # is this value is positive, it means that this system has a higher
        # average accuracy value -> is better
        "mean_difference_b_minus_a": measurements_b.mean() - measurements_a.mean(),
        "p_a_better": probability_a,
        "p_practical_tie": probability_tie,
        "p_b_better": probability_b,
        # the effect size etsimated by the model
        "effect_size": posterior.effect_size,
    }


def compare_all_pairs(metrics, metric="accuracy", rope_size=0.005):
    """
    Compares all unique pairs, results are saved in rope_comparisons.csv.
    """
    systems = sorted(set(zip(metrics["representation"],
                             metrics["architecture"])))

    rows = []

    for (representation_a, architecture_a), (representation_b, architecture_b,
                                             ) in itertools.combinations(systems, 2):
        rows.append(compare_pair(metrics,
                                 representation_a,
                                 architecture_a,
                                 representation_b,
                                 architecture_b,
                                 metric=metric,
                                 rope_size=rope_size))

    return pd.DataFrame(rows)

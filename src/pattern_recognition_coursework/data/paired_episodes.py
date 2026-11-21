"""Paired one/five-shot plans using the existing sampler and plan validator."""
import copy

from pattern_recognition_coursework.data.cub_dataset import FewShotBatchSampler
from pattern_recognition_coursework.data.episodes import dataset_fingerprint, validate_plan


def make_paired_plans(dataset, episodes=600, seed=42):
    if type(episodes) is not int or episodes < 1 or type(seed) is not int or seed < 0:
        raise ValueError('episodes must be positive and seed must be nonnegative integers')
    five = dict(schema='few-shot-episodes-v1', dataset_fingerprint=dataset_fingerprint(dataset),
                n_way=5, k_shot=5, n_query=5, episodes=episodes, seed=seed,
                batches=list(FewShotBatchSampler(dataset, episodes, 5, 5, 5, seed=seed)))
    one = copy.deepcopy(five)
    one['k_shot'] = 1
    # First of each class's five support images; all 25 queries remain unchanged.
    one['batches'] = [batch[:25:5]+batch[25:] for batch in five['batches']]
    validate_paired_plans(dataset, one, five)
    return {1: one, 5: five}


def validate_paired_plans(dataset, one, five):
    if one['seed'] != five['seed'] or one['episodes'] != five['episodes']:
        raise ValueError('Paired plans must have the same seed and episode count')
    for k, plan in [(1, one), (5, five)]:
        validate_plan(plan, dataset, 5, k, 5, five['episodes'])
    for a, b in zip(one['batches'], five['batches']):
        if a[:5] != b[:25:5] or a[5:] != b[25:]:
            raise ValueError('Paired shots must share classes/queries and nested support')

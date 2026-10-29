"""Evaluation-only, in-memory goal normalization intervention."""
import numpy as np
import torch


@torch.no_grad()
def replace_goal_statistics(policy, target, donor):
    if target['signature'] != donor['signature'] or target['identity'] != donor['identity']:
        raise ValueError('Donor configuration/assets differ from target')
    for key in ('mean', 'components'):
        np.testing.assert_array_equal(target['pca'][key], donor['pca'][key])
    before = {key: value.clone() for key, value in policy.state_dict().items()}
    report = {'target_transitions': target['learner']['transitions'],
              'donor_transitions': donor['learner']['transitions'], 'statistics': {}}
    allowed = {f'normalizer.state.{field}' for field in ('mean', 'var', 'count')}
    for field in ('mean', 'var', 'count'):
        key = f'normalizer.state.{field}'
        values = donor['learner']['policy'][key][:2]
        if not torch.isfinite(values).all() or (field != 'mean' and not (values > 0).all()):
            raise ValueError(f'Invalid donor goal {field}')
        report['statistics'][field] = dict(before=before[key][:2].cpu().tolist(), after=values.cpu().tolist())
        getattr(policy.normalizer.state, field)[:2].copy_(values)
    for key, value in policy.state_dict().items():
        if key in allowed:
            torch.testing.assert_close(value[2:], before[key][2:], rtol=0, atol=0)
        else:
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    return report

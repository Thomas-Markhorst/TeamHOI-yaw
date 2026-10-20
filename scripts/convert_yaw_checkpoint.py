"""Produce an inference-only yaw checkpoint; training uses --init_from_checkpoint."""
import argparse
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'teamhoi'))
from utils.yaw_checkpoint import expand_yaw_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source')
    parser.add_argument('destination')
    args = parser.parse_args()
    destination = Path(args.destination)
    if destination.exists():
        parser.error('Destination already exists; refusing to overwrite it')
    checkpoint = torch.load(args.source, map_location='cpu')
    target = dict(checkpoint['model'])
    for key in ('a2c_network.task_encoder.1.0.weight',
                'a2c_network.task_encoder_critic.1.0.weight'):
        weight = target[key]
        target[key] = weight.new_zeros((weight.shape[0], 7))
    stats_target = {key: checkpoint['running_mean_std'][key].new_zeros(
        (checkpoint['running_mean_std'][key].numel() + 4,))
        for key in ('running_mean', 'running_var')}
    model, stats = expand_yaw_checkpoint(checkpoint, target, stats_target)
    result = {key: checkpoint[key] for key in ('amp_input_mean_std', 'amp_masked_input_mean_std')}
    result.update(model=model, running_mean_std=stats,
                  yaw_conversion=dict(source=str(Path(args.source)), inference_only=True))
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, destination)
    print('Saved inference-only yaw checkpoint to {}'.format(destination))


if __name__ == '__main__':
    main()

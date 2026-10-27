"""Destination-only yaw reward comparisons; no simulator required."""
import math
from pathlib import Path
import sys
import unittest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'teamhoi'))
from utils.yaw import (yaw_settings, rotation_reward, rotation_reward_only_error,
                       rotation_reward_only_error_drift, control_heading_error,
                       transport_rotation_reward)


class RewardVariantTests(unittest.TestCase):
    def cfg(self, variant):
        return yaw_settings(dict(enabled=True,reward_variant=variant,position_weight=.3))

    def reward(self, variant, error=.2, speed=0., distance=0., arrived=True, put=False, support=True):
        return rotation_reward(torch.tensor([error]),torch.tensor([speed]),torch.tensor([distance]),
            torch.tensor([arrived]),torch.tensor([put]),torch.tensor([support]),self.cfg(variant)).item()

    def test_training_entrypoint_dispatches_to_both_user_functions(self):
        args=(torch.tensor([.2]),torch.tensor([.8]),torch.tensor([.1]),
              torch.tensor([True]),torch.tensor([False]),torch.tensor([True]))
        for variant,fn in [('only_error',rotation_reward_only_error),('only_error_drift',rotation_reward_only_error_drift)]:
            cfg=self.cfg(variant)
            torch.testing.assert_close(rotation_reward(*args,cfg),fn(*args,cfg))
        with self.assertRaises(ValueError): yaw_settings({'reward_variant':'typo'})

    def test_error_only_has_no_distance_or_speed_reward(self):
        value=self.reward('only_error')
        self.assertAlmostEqual(value,1-.2/math.pi,places=6)
        self.assertEqual(value,self.reward('only_error',distance=2.,speed=3.))
        self.assertGreater(value,self.reward('only_error',error=.8))

    def test_positive_position_term_has_requested_weight_and_scale(self):
        base=self.reward('only_error')
        for d in [0.,.03,.1,.5]:
            value=self.reward('only_error_drift',distance=d)
            self.assertAlmostEqual(value-base,.3*math.exp(-20*d*d),places=6)
        self.assertGreater(self.reward('only_error_drift',distance=.03),self.reward('only_error_drift',distance=.1))
        self.assertEqual(self.reward('only_error_drift',speed=0.),self.reward('only_error_drift',speed=3.))

    def test_arrival_support_and_release_gates_preserved(self):
        for variant in ['only_error','only_error_drift']:
            self.assertEqual(self.reward(variant,arrived=False),0.)
            self.assertEqual(self.reward(variant,support=False),0.)
            self.assertEqual(self.reward(variant),self.reward(variant,put=True,support=False))

    def test_configs_keep_adapted_yaw_task_without_transport_heading(self):
        base=yaml.safe_load((ROOT/'cfg/teamhoi_yaw.yaml').read_text())['params']
        for variant in ['only_error','only_error_drift']:
            new=yaml.safe_load((ROOT/'cfg'/('teamhoi_yaw_'+variant+'.yaml')).read_text())['params']
            self.assertEqual(base['network'],new['network'])
            self.assertEqual(base['config']['w'],new['config']['w'])
            cfg=yaw_settings(new['config']['yaw'])
            self.assertEqual(cfg['reward_variant'],variant)
            self.assertEqual(cfg['position_weight'],.3)
            self.assertFalse(cfg['staged_transport'])
            self.assertEqual(cfg['in_place_probability'],0.)
            self.assertEqual(cfg['carry_heading_weight'],0.)
            self.assertEqual(cfg['carry_angular_speed_weight'],0.)
            cur=torch.tensor([.2]);target=torch.tensor([.7]);pickup=torch.tensor([.1]);arr=torch.tensor([False])
            torch.testing.assert_close(control_heading_error(cur,target,pickup,arr,cfg),target-cur)
            torch.testing.assert_close(transport_rotation_reward(cur,cur,~arr,arr,cfg),torch.zeros(1))
            for key in ['initial_yaw_range_deg','target_offset_range_deg','reference_loss_coef',
                        'position_tolerance','yaw_tolerance_deg','angular_speed_tolerance']:
                self.assertEqual(cfg[key],base['config']['yaw'][key])

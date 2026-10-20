"""CPU checks: python -m unittest discover -s tests -v (no Isaac Gym required)."""
import ast
import copy
import math
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'teamhoi'))
from utils.yaw import (yaw_settings, wrap_angle, quaternion_yaw, yaw_quaternion,
                       yaw_features, update_yaw_phase, rotation_reward, placement_metrics)
from utils.yaw_checkpoint import expand_yaw_checkpoint
from utils.yaw_reference import build_reference, reference_loss


def task_method(name, **extra):
    # Execute the real task method without importing Isaac Gym's native runtime.
    path = ROOT / 'teamhoi/env/tasks/humanoid_multi_amp_teamhoi_task.py'
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    env = dict(torch=torch, YAW_EXTRA_DIM=4, wrap_angle=wrap_angle,
               quaternion_yaw=quaternion_yaw, yaw_features=yaw_features,
               update_yaw_phase=update_yaw_phase)
    env.update(extra)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), env)
    return env[name]


class YawTaskTests(unittest.TestCase):
    def setUp(self):
        self.cfg = yaw_settings({'enabled': True})

    def test_wrap_and_exact_heading(self):
        e = wrap_angle(torch.deg2rad(torch.tensor([-358., 358., 180., 360.])))
        torch.testing.assert_close(e.abs(), torch.deg2rad(torch.tensor([2., 2., 180., 0.])), atol=1e-6, rtol=1e-5)
        angle = torch.tensor([-3., -1., 0., 2.5])
        q = yaw_quaternion(angle)
        torch.testing.assert_close(quaternion_yaw(q), angle)
        torch.testing.assert_close(quaternion_yaw(-q), angle)

    def test_phase_latches_and_braking_gate(self):
        arrived = put = torch.tensor([False])
        def phase(d, deg, speed):
            return update_yaw_phase(arrived, put, torch.tensor([d]),
                                    torch.tensor([math.radians(deg)]), torch.tensor([speed]), self.cfg)
        arrived, put = phase(.031, 20, 0.)
        self.assertFalse(arrived.item())
        arrived, put = phase(.029, 20, 0.)
        self.assertTrue(arrived.item()); self.assertFalse(put.item())
        arrived, put = phase(.05, 0., 0.)
        self.assertTrue(arrived.item()); self.assertFalse(put.item())
        arrived, put = phase(.02, 4., .2)
        self.assertFalse(put.item())
        arrived, put = phase(.02, 4., .01)
        self.assertTrue(put.item())
        arrived, put = phase(.2, 50., .5)
        self.assertTrue(put.item())

    def test_reward_activation_and_damping(self):
        def reward(error, speed=0., arrived=True, put=False, support=True):
            return rotation_reward(torch.tensor([error]), torch.tensor([speed]), torch.tensor([.02]),
                                   torch.tensor([arrived]), torch.tensor([put]), torch.tensor([support]), self.cfg).item()
        self.assertEqual(reward(1., arrived=False), 0.)
        self.assertEqual(reward(1., put=True), 0.)
        self.assertEqual(reward(1., support=False), 0.)
        self.assertGreater(reward(.1), reward(.5))
        self.assertLess(reward(.1, .5), reward(.1, 0.))
        self.assertGreater(reward(.1)-reward(.1, .5), reward(1.)-reward(1., .5))

    def test_settled_success_is_not_historical(self):
        state = torch.zeros(1, 13); state[:, 6] = 1; state[:, 2] = .82
        target = torch.zeros(1, 3); q = yaw_quaternion(torch.zeros(1))
        hands = torch.ones(1, 3, 2)
        active = torch.tensor([[True, True, False]])
        put = torch.ones(1, dtype=torch.bool)
        count = torch.zeros(1, dtype=torch.long)
        for _ in range(self.cfg['settled_steps']):
            _, _, _, count, success = placement_metrics(state, target, q, hands, active, put, count, self.cfg)
        self.assertTrue(success.item())
        state[:, 3:7] = yaw_quaternion(torch.tensor([.2]))
        _, _, _, count, success = placement_metrics(state, target, q, hands, active, put, count, self.cfg)
        self.assertFalse(success.item()); self.assertEqual(count.item(), 0)
        state[:, 3:7] = q; hands[0, 0] = 0.
        _, _, _, _, success = placement_metrics(state, target, q, hands, active, put, count, self.cfg)
        self.assertFalse(success.item())

    def test_subset_observations_do_not_corrupt_other_environments(self):
        def compute_obs(roots, pos, rim, vec, target, hand_z, putdown, num_humanoids):
            # Stand in for the unchanged legacy geometric feature extractor.
            out = roots.new_zeros((roots.shape[0], 204 + 9 * num_humanoids))
            out[:, 201:204] = target
            return out
        method = task_method('_compute_task_obs_tensor', compute_obs=compute_obs)
        t = types.SimpleNamespace(device='cpu', num_envs=4, num_agents=2,
            yaw_enabled=True, yaw_config=self.cfg, viz=False, log_metrics=False)
        t._humanoid_root_states_tensor = torch.zeros(4, 2, 13)
        t._obj_states = torch.zeros(4, 13); t._obj_states[:, 6] = 1
        t._target_rot = yaw_quaternion(torch.tensor([.3, 0., .3, .5]))
        t._target_end_pos = torch.tensor([[2., 0., .94], [.02, 0., .94], [2., 0., .94], [.02, 0., .94]])
        t._rigid_body_pos_tensor = torch.zeros(4, 2, 2, 3)
        t._lift_body_ids = torch.tensor([0, 1]); t.obj_held_points = torch.zeros(4, 64, 3)
        t.putdown_mask = torch.zeros(4, dtype=torch.bool); t.arrived_mask = t.putdown_mask.clone()
        t.task_obs_dim = [3, 192, 6, 7, 18]
        t.update_held_points = lambda *_: None
        before = t._target_end_pos.clone()
        obs = method(t, torch.tensor([3, 1]))
        self.assertEqual(tuple(obs.shape), (2, 2, 226))
        self.assertEqual(t.arrived_mask.tolist(), [False, True, False, True])
        self.assertEqual(t.putdown_mask.tolist(), [False, True, False, False])
        torch.testing.assert_close(t._target_end_pos, before)
        torch.testing.assert_close(obs[0, 0, 204:208], yaw_features(torch.tensor(.5), torch.tensor(0.), torch.tensor(True), 1.))
        self.assertAlmostEqual(obs[1, 0, 203].item(), .78, places=5)
        # Repeated calls and a reset of one environment preserve all other flags.
        t.arrived_mask[1] = False; t.putdown_mask[1] = False
        t._target_end_pos[1, 0] = 2.
        method(t, torch.tensor([1]))
        self.assertFalse(t.arrived_mask[1]); self.assertTrue(t.arrived_mask[3])
        obs = method(t)
        self.assertEqual(obs.shape[0], 4)

    def test_initial_and_target_yaw_randomization_respects_subset(self):
        import numpy as np
        method = task_method('_reset_obj', np=np, yaw_quaternion=yaw_quaternion)
        t = types.SimpleNamespace(device='cpu', yaw_enabled=True, yaw_config=self.cfg,
                                  obj_height=.82, _obj_states=torch.randn(1000,13),
                                  _target_rot=yaw_quaternion(torch.zeros(1000)))
        before=t._obj_states.clone(); ids=torch.arange(0,1000,2)
        method(t,ids)
        initial=quaternion_yaw(t._obj_states[ids,3:7])
        offset=wrap_angle(quaternion_yaw(t._target_rot[ids])-initial)
        self.assertLessEqual(initial.abs().max().item(),math.radians(5))
        self.assertLessEqual(offset.abs().max().item(),math.radians(30))
        self.assertTrue((offset>0).any()); self.assertTrue((offset<0).any())
        torch.testing.assert_close(t._obj_states[1::2],before[1::2])
        self.assertEqual(t._obj_states[ids,7:].abs().sum().item(),0.)

    def test_configuration_validation(self):
        for config in ({'position_tolerance': 0}, {'target_offset_range_deg': 181}, {'typo': True}):
            with self.assertRaises(ValueError): yaw_settings(config)


class CheckpointTests(unittest.TestCase):
    def test_copy_trained_columns_and_reject_wrong_layout(self):
        k = 'a2c_network.task_encoder.1.0.weight'
        c = 'a2c_network.task_encoder_critic.1.0.weight'
        old = {k:torch.randn(5,3), c:torch.randn(5,3), 'bias':torch.randn(5)}
        target = {k:torch.zeros(5,7), c:torch.zeros(5,7), 'bias':torch.zeros(5)}
        checkpoint = {'model':old, 'running_mean_std':{'running_mean':torch.randn(427),
                       'running_var':torch.ones(427), 'count':torch.tensor(10.)}}
        model, stats = expand_yaw_checkpoint(checkpoint, target, {'running_mean':torch.zeros(431), 'running_var':torch.ones(431)})
        for key in (k,c):
            torch.testing.assert_close(model[key][:,:3], old[key]); self.assertEqual(model[key][:,3:].abs().sum(), 0)
        torch.testing.assert_close(stats['running_mean'][:427], checkpoint['running_mean_std']['running_mean'])
        self.assertEqual(stats['running_var'][427:].tolist(), [1.]*4)
        with self.assertRaises(ValueError): expand_yaw_checkpoint({'model':model, 'running_mean_std':stats},target,stats)

    @unittest.skipUnless(os.environ.get('TEAMHOI_BASE_CHECKPOINT'), 'set TEAMHOI_BASE_CHECKPOINT for real checkpoint parity')
    def test_real_checkpoint_actor_critic_and_reference(self):
        from rl_games.algos_torch.running_mean_std import RunningMeanStd
        from learning.transformer.teamhoi_transformer import AMPTransformerMultiTaskBuilder
        checkpoint = torch.load(os.environ['TEAMHOI_BASE_CHECKPOINT'], map_location='cpu')
        params = yaml.safe_load((ROOT/'cfg/teamhoi.yaml').read_text())['params']['network']
        def network(goal, n):
            return AMPTransformerMultiTaskBuilder.UnifiedNetworkClass(params, self_obs_size=223,
                task_obs_size=201+goal+9*n, goal_obs_size=[3,192,6,goal], others_obs_each_size=9,
                num_agents=n, system_max_agents=8, device='cpu', actions_num=28,
                input_shape=(424+goal+9*n,), num_seqs=n, value_size=1, amp_input_shape=(625,), goal_multiplier=1)
        for n in (2,4,8):
            with self.subTest(team_size=n):
                old = network(3,n).eval(); new = network(7,n).eval()
                old.load_state_dict({k[len('a2c_network.'):]:v for k,v in checkpoint['model'].items()},strict=True)
                target = {'a2c_network.'+k:v for k,v in new.state_dict().items()}
                model, stats = expand_yaw_checkpoint(checkpoint,target, {'running_mean':torch.zeros(431),'running_var':torch.ones(431)})
                new.load_state_dict({k[len('a2c_network.'):]:v for k,v in model.items()},strict=True)
                old_norm=RunningMeanStd((427,)).eval(); old_norm.load_state_dict(checkpoint['running_mean_std'])
                new_norm=RunningMeanStd((431,)).eval(); new_norm.load_state_dict(stats)
                old_obs=torch.randn(n*2,427+9*n)
                new_obs=torch.cat((old_obs[:,:427],torch.randn(n*2,4),old_obs[:,427:]),-1)
                new_obs[:,430]=0
                mask=torch.ones(n*2,dtype=torch.bool); mask[-1]=False
                mask_others=torch.ones(n*2,n-1,dtype=torch.bool)
                mask_others[-n:,-1]=False
                old_proc=old_obs.clone(); old_proc[:,:427]=old_norm(old_proc[:,:427])
                new_proc=new_obs.clone(); new_proc[:,:431]=new_norm(new_proc[:,:431])
                with torch.no_grad():
                    old_mu,_=old.eval_actor(old_proc,mask,mask_others)
                    new_mu,_=new.eval_actor(new_proc,mask,mask_others)
                    torch.testing.assert_close(new_mu,old_mu,atol=2e-6,rtol=2e-5)
                    torch.testing.assert_close(new.eval_critic(new_proc,mask,mask_others),old.eval_critic(old_proc,mask,mask_others),atol=2e-6,rtol=2e-5)
                a=types.SimpleNamespace(model=types.SimpleNamespace(a2c_network=new), running_mean_std=new_norm,
                                        normalize_input=True,self_obs_size=431)
                build_reference(a)
                self.assertAlmostEqual(reference_loss(a,new_obs,mask,mask_others,new_mu).item(),0.,places=6)
                self.assertGreater(reference_loss(a,new_obs,mask,mask_others,new_mu+.1).item(),0.)
                saved=copy.deepcopy({'model':a._yaw_reference.state_dict(),
                                     'normalization':a._yaw_reference_norm.state_dict()})
                with torch.no_grad(): new.composer.actors[-1].bias.add_(.1)
                build_reference(a,saved)
                # A resumed teacher must remain the original actor, not the changed student.
                self.assertAlmostEqual(reference_loss(a,new_obs,mask,mask_others,new_mu).item(),0.,places=6)
                prediction,_=new.eval_actor(new_proc,mask,mask_others)
                preserve=reference_loss(a,new_obs,mask,mask_others,prediction)
                self.assertGreater(preserve.item(),0.)
                preserve.backward()
                self.assertIsNotNone(new.composer.actors[-1].bias.grad)
                self.assertTrue(all(p.grad is None for p in a._yaw_reference.parameters()))
                new_obs[:,430]=1
                self.assertEqual(reference_loss(a,new_obs,mask,mask_others,new_mu+.1).item(),0.)


if __name__=='__main__':
    unittest.main()

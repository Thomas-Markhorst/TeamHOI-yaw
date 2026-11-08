"""Regression checks for the command sequence and mixed-team configurations."""
import math
import types
import unittest

import torch
import yaml

from test_yaw import ROOT, task_method
from utils.pose_control import POSE_EXTRA_DIM, pose_features, pose_rotation_reward, pose_settings
from utils.yaw import yaw_quaternion, yaw_settings


class SequentialYawTests(unittest.TestCase):
    def test_command_switch_and_original_vertical_cue(self):
        def geometry(roots, pos, rim, vec, target, hand_z, putdown, num_humanoids, expose_height=False):
            self.assertFalse(expose_height)
            obs = roots.new_zeros((roots.shape[0], 204 + 9*num_humanoids))
            obs[:, 203] = torch.where(putdown.repeat_interleave(num_humanoids), target[:, 2], 0.)
            return obs

        method = task_method('_compute_task_obs_tensor', compute_obs=geometry,
                             pose_features=pose_features, POSE_EXTRA_DIM=POSE_EXTRA_DIM, yaw_quaternion=yaw_quaternion)
        n, a = 3, 4
        cfg = pose_settings(dict(enabled=True, sequential_agents=4))
        current = yaw_quaternion(torch.tensor([.2, .2, .2]))
        target = yaw_quaternion(torch.tensor([.5, .5, .5]))
        t = types.SimpleNamespace(
            device='cpu', num_envs=n, num_agents=a, yaw_enabled=True, pose_enabled=True,
            pose_config=cfg, yaw_config=yaw_settings({}), viz=False, log_metrics=False,
            _humanoid_root_states_tensor=torch.zeros(n,a,13), _obj_states=torch.zeros(n,13),
            _target_rot=target, _target_end_pos=torch.tensor([[2.,0.,.94],[.02,0.,.94],[.02,0.,.94]]),
            _rigid_body_pos_tensor=torch.zeros(n,a,2,3), _lift_body_ids=torch.tensor([0,1]),
            obj_held_points=torch.zeros(n,64,3), task_obs_dim=[3,192,6,12,36],
            putdown_mask=torch.zeros(n,dtype=torch.bool), arrived_mask=torch.zeros(n,dtype=torch.bool),
            carry_yaw=torch.tensor([.1,.1,.1]), lifted_mask=torch.zeros(n,dtype=torch.bool),
            update_held_points=lambda *_: None)
        t._obj_states[:,3:7] = current
        t._obj_states[2,3:7] = target[2]
        obs = method(t)
        self.assertEqual(t.arrived_mask.tolist(), [False,True,True])
        self.assertEqual(t.putdown_mask.tolist(), [False,False,True])
        torch.testing.assert_close(obs[:,0,206], torch.sin(torch.tensor([-.1,.3,0.])), atol=1e-6,rtol=0)
        torch.testing.assert_close(obs[:,0,203], torch.tensor([0.,0.,.78]))
        torch.testing.assert_close(t._target_rot,target)
        # Once arrival latches, drifting back outside 3 cm does not restore the initial heading.
        t._obj_states[1,0] = .3
        obs = method(t,torch.tensor([1]))
        self.assertAlmostEqual(obs[0,0,206].item(), math.sin(.3), places=6)
        self.assertFalse(t.putdown_mask[1])

    def test_pose_reward_keeps_alignment_incentive_through_release(self):
        cfg=pose_settings(dict(enabled=True,sequential_agents=4))
        reward=pose_rotation_reward(torch.zeros(2,3),torch.zeros(2),
                                    torch.zeros(2,dtype=torch.bool),cfg,
                                    torch.tensor([False,True]))
        torch.testing.assert_close(reward,torch.tensor([0.,1.]))

    def test_configs_match_except_team_size_and_output_name(self):
        configs=[]
        for name,team in [('four',4),('mixed',0)]:
            c=yaml.safe_load((ROOT/'cfg'/('teamhoi_sequential_yaw_'+name+'.yaml')).read_text())['params']['config']
            self.assertEqual(c['pose_control'].pop('sequential_agents'),team)
            self.assertEqual(c['pose_control']['walking_probability'],1.)
            self.assertEqual(c['pose_control']['initial_yaw_range_deg'],5.)
            self.assertEqual(yaw_settings(c['yaw'])['target_offset_range_deg'],30.)
            self.assertEqual(yaw_settings(c['yaw'])['in_place_probability'],0.)
            self.assertLess(c['minibatch_size']*8,65536)
            self.assertEqual(c['w'][9],1.)
            c.pop('name')
            configs.append(c)
        self.assertEqual(*configs)

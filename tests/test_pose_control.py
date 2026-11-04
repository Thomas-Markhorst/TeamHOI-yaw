"""Pose command regression tests, including real task methods without Isaac Gym."""
import ast
import math
import os
from pathlib import Path
import sys
import types
import unittest
import torch
import yaml
from test_yaw import ROOT, task_method
from utils.pose_control import (pose_settings, axis_quaternion, quat_mul, quat_conjugate,
    rotate_inverse, local_orientation_error, pose_features, sample_pose_targets,
    pose_rotation_reward, pose_hold_metrics)
from utils.yaw import yaw_settings, yaw_quaternion
from utils.yaw_checkpoint import prepare_yaw_checkpoint

class PoseTests(unittest.TestCase):
    def test_local_axis_commands_are_invariant_to_world_heading(self):
        for axis in (0,1,2):
            angle=torch.tensor([.4]);local=axis_quaternion(angle,axis)
            for heading in (-2.,-.4,.7854,2.8):
                current=yaw_quaternion(torch.tensor([heading]))
                target=quat_mul(current,local)
                expected=torch.zeros(1,3);expected[:,axis]=angle
                torch.testing.assert_close(local_orientation_error(current,target),expected,atol=1e-6,rtol=1e-5)
                omega=torch.tensor([[.2,.3,.4]])
                world=rotate_inverse(quat_conjugate(current),omega)
                features=pose_features(current,target,world)
                torch.testing.assert_close(features[:,-3:],omega,atol=1e-6,rtol=1e-5)
                torch.testing.assert_close(pose_features(current,-target,world),features,atol=1e-6,rtol=1e-5)

    def test_sampling_single_local_axis_and_walking_mixture(self):
        n=20000;torch.manual_seed(42)
        initial=yaw_quaternion(torch.rand(n)*6-3)
        start=torch.randn(n,2);walking=start+4
        for mode in ('yaw','pitch_roll'):
            cfg=pose_settings(dict(enabled=True,mode=mode))
            xy,target,in_place=sample_pose_targets(initial,start,walking,cfg)
            self.assertTrue(.14 < (~in_place).float().mean() < .16)
            torch.testing.assert_close(xy[in_place],start[in_place])
            torch.testing.assert_close(xy[~in_place],walking[~in_place])
            error=local_orientation_error(initial,target)
            torch.testing.assert_close(error[~in_place],torch.zeros_like(error[~in_place]),atol=1e-6,rtol=0)
            changed=error[in_place].abs()>1e-5
            self.assertTrue((changed.sum(-1)==1).all())
            if mode=='yaw':self.assertTrue(changed[:,2].all())
            else:self.assertFalse(changed[:,2].any());self.assertTrue(changed[:,0].any());self.assertTrue(changed[:,1].any())
            magnitude=error[in_place].abs().max(-1).values
            self.assertGreaterEqual(magnitude.min().item(),math.radians(5)-1e-6)
            self.assertLessEqual(magnitude.max().item(),math.radians(30)+1e-6)

    def test_reward_has_no_arrival_distance_or_velocity_gate(self):
        cfg=pose_settings({'enabled':True});height=torch.tensor([.95]);yes=torch.tensor([True])
        def reward(e,h=height,s=yes):return pose_rotation_reward(torch.tensor([e]),h,s,.82,cfg).item()
        self.assertAlmostEqual(reward([0.,0.,.3]),1-.3/math.pi,places=6)
        # A single-axis tilt uses pi/2, without dilution by averaging axes.
        self.assertAlmostEqual(reward([.3,0.,0.]),1-.3/(math.pi/2),places=6)
        self.assertGreater(reward([.1,0.,0.]),reward([.3,0.,0.]))
        self.assertEqual(reward([0.,0.,0.],s=~yes),0.)
        self.assertEqual(reward([0.,0.,0.],h=torch.tensor([.85])),0.)

    def test_hold_accepts_tilt_and_variable_height_but_requires_support(self):
        cfg=pose_settings({'enabled':True});q=axis_quaternion(torch.tensor([.5]),0)
        pos=torch.tensor([[0.,0.,1.05]]);omega=torch.zeros(1,3);yes=torch.tensor([True]);count=torch.zeros(1,dtype=torch.long)
        for _ in range(15):count,success,_,_=pose_hold_metrics(q,q,pos,pos,omega,yes,.82,count,cfg)
        self.assertTrue(success.item())
        for badpos,badomega,support in [(pos+torch.tensor([[.05,0.,0.]]),omega,yes),(pos,omega+.2,yes),(pos,omega,~yes)]:
            count,success,_,_=pose_hold_metrics(q,q,badpos,pos,badomega,support,.82,count,cfg)
            self.assertFalse(success.item());self.assertEqual(count.item(),0)

    def test_real_subset_observations_have_no_phase_flag_or_auto_putdown(self):
        def geom(roots,pos,rim,vec,target,hand_z,putdown,num_humanoids,expose_height=False):
            self.assertTrue(expose_height)
            return roots.new_zeros((roots.shape[0],204+9*num_humanoids))
        method=task_method('_compute_task_obs_tensor',compute_obs=geom,pose_features=pose_features,POSE_EXTRA_DIM=9)
        n,a=3,4
        t=types.SimpleNamespace(device='cpu',num_envs=n,num_agents=a,yaw_enabled=True,pose_enabled=True,
          pose_config=pose_settings({'enabled':True}),yaw_config=yaw_settings({}),viz=False,log_metrics=False,
          _humanoid_root_states_tensor=torch.zeros(n,a,13),_obj_states=torch.zeros(n,13),
          _target_rot=yaw_quaternion(torch.ones(n)*.3),_target_end_pos=torch.zeros(n,3),
          _rigid_body_pos_tensor=torch.zeros(n,a,2,3),_lift_body_ids=torch.tensor([0,1]),
          obj_held_points=torch.zeros(n,64,3),task_obs_dim=[3,192,6,12,36],
          putdown_mask=torch.zeros(n,dtype=torch.bool),arrived_mask=torch.zeros(n,dtype=torch.bool),
          update_held_points=lambda *_:None)
        t._obj_states[:,6]=1
        obs=method(t,torch.tensor([2,0]))
        self.assertEqual(tuple(obs.shape),(2,4,249))
        self.assertFalse(t.putdown_mask.any());self.assertFalse(t.arrived_mask.any())
        torch.testing.assert_close(obs[0,0,204:213],pose_features(t._obj_states[2:3,3:7],t._target_rot[2:3],torch.zeros(1,3))[0])

    def test_pose_configs_keep_original_learning_losses_and_four_agent_contract(self):
        for mode in ('yaw','pitch_roll'):
            cfg=yaml.safe_load((ROOT/'cfg'/('teamhoi_pose_'+mode+'.yaml')).read_text())['params']['config']
            self.assertEqual(cfg['yaw']['reference_loss_coef'],0.)
            self.assertEqual(cfg['w'][8],0.);self.assertEqual(cfg['w'][9],1.)
            self.assertEqual(cfg['w'][7],1.)
            self.assertEqual(pose_settings(cfg['pose_control'])['walking_probability'],.15)
            self.assertEqual(cfg['max_epochs'],3999)

    def test_real_hand_reward_keeps_separation_and_removes_equal_height_only(self):
        path=ROOT/'teamhoi/env/tasks/reward_functions.py'
        fn=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='compute_hands_and_lifts_reward')
        fn.decorator_list=[];ns={'torch':torch}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),ns)
        reward=ns[fn.name]
        hands=torch.tensor([[[0.,0.,1.],[.433,0.,1.25]]]);rim=hands.clone()
        original=reward(hands,rim,torch.zeros(1))[0]
        pose=reward(hands,rim,torch.zeros(1),pose_control=True)[0]
        self.assertAlmostEqual((original/pose).item(),math.exp(-20*.25**2),places=6)
        self.assertAlmostEqual(pose.item(),1.,places=6)

    @unittest.skipUnless(os.environ.get('TEAMHOI_BASE_CHECKPOINT'),'real checkpoint optional')
    def test_real_checkpoint_actor_and_critic_parity_with_twelve_goal_inputs(self):
        from rl_games.algos_torch.running_mean_std import RunningMeanStd
        from learning.transformer.teamhoi_transformer import AMPTransformerMultiTaskBuilder
        checkpoint=torch.load(os.environ['TEAMHOI_BASE_CHECKPOINT'],map_location='cpu')
        params=yaml.safe_load((ROOT/'cfg/teamhoi.yaml').read_text())['params']['network']
        def network(goal):
            return AMPTransformerMultiTaskBuilder.UnifiedNetworkClass(params,self_obs_size=223,
              task_obs_size=201+goal+36,goal_obs_size=[3,192,6,goal],others_obs_each_size=9,
              num_agents=4,system_max_agents=8,device='cpu',actions_num=28,input_shape=(424+goal+36,),
              num_seqs=4,value_size=1,amp_input_shape=(625,),goal_multiplier=1)
        old=network(3).eval();new=network(12).eval()
        old.load_state_dict({k[len('a2c_network.'):]:v for k,v in checkpoint['model'].items()})
        model,stats=prepare_yaw_checkpoint(checkpoint,{'a2c_network.'+k:v for k,v in new.state_dict().items()},
                                          {'running_mean':torch.zeros(436),'running_var':torch.ones(436)})
        new.load_state_dict({k[len('a2c_network.'):]:v for k,v in model.items()})
        oldnorm=RunningMeanStd((427,)).eval();oldnorm.load_state_dict(checkpoint['running_mean_std'])
        newnorm=RunningMeanStd((436,)).eval();newnorm.load_state_dict(stats)
        obs=torch.randn(8,463);expanded=torch.cat((obs[:,:427],torch.randn(8,9),obs[:,427:]),-1)
        obs[:,:427]=oldnorm(obs[:,:427]);expanded[:,:436]=newnorm(expanded[:,:436])
        mask=torch.ones(8,dtype=torch.bool);others=torch.ones(8,3,dtype=torch.bool)
        with torch.no_grad():
            torch.testing.assert_close(old.eval_actor(obs,mask,others)[0],new.eval_actor(expanded,mask,others)[0],atol=2e-6,rtol=2e-5)
            torch.testing.assert_close(old.eval_critic(obs,mask,others),new.eval_critic(expanded,mask,others),atol=2e-6,rtol=2e-5)

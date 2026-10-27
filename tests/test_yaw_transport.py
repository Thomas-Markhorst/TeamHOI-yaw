"""Transport heading and pickup-first rotation tests, without a native simulator."""
import math
from pathlib import Path
import types
import unittest
import torch
import yaml
from test_yaw import ROOT, task_method
from utils.yaw import (yaw_settings, yaw_quaternion, update_yaw_phase, update_pickup,
                       control_heading_error, transport_rotation_reward, sample_in_place_targets)


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.cfg = yaw_settings(dict(enabled=True, staged_transport=True,
            carry_heading_weight=1., carry_angular_speed_weight=.1))

    def test_in_place_requires_pickup_even_if_yaw_already_matches(self):
        f = torch.tensor([False]); z = torch.zeros(1)
        a,p=update_yaw_phase(f,f,z,z,z,self.cfg,f)
        self.assertFalse(a.item()); self.assertFalse(p.item())
        a,p=update_yaw_phase(a,p,z,z,z,self.cfg,~f)
        self.assertTrue(a.item()); self.assertTrue(p.item())

    def test_capture_heading_once_and_only_after_supported_upright_lift(self):
        f=torch.tensor([False]); ref=torch.tensor([.2]); cur=torch.tensor([.3])
        for h,s,u in [(.85,True,True),(.94,False,True),(.94,True,False)]:
            lifted,head=update_pickup(f,ref,cur,torch.tensor([h]),torch.tensor([s]),torch.tensor([u]),.82,self.cfg)
            self.assertFalse(lifted.item()); torch.testing.assert_close(head,ref)
        lifted,head=update_pickup(f,ref,cur,torch.tensor([.94]),~f,~f,.82,self.cfg)
        self.assertTrue(lifted.item()); torch.testing.assert_close(head,cur)
        _,head=update_pickup(lifted,head,torch.tensor([.8]),torch.tensor([.94]),~f,~f,.82,self.cfg)
        torch.testing.assert_close(head,cur)

    def test_heading_observation_switch_and_reward_do_not_fight_final_rotation(self):
        current=torch.tensor([.2,.2]); target=torch.tensor([1.,1.]); pickup=torch.tensor([.1,.1])
        arrived=torch.tensor([False,True])
        error=control_heading_error(current,target,pickup,arrived,self.cfg)
        torch.testing.assert_close(error,torch.tensor([-.1,.8]))
        def reward(err,speed,lifted=True,arr=False):
            return transport_rotation_reward(torch.tensor([err]),torch.tensor([speed]),torch.tensor([lifted]),torch.tensor([arr]),self.cfg).item()
        self.assertEqual(reward(0.,0.),0.)
        self.assertLess(reward(.2,0.),reward(.1,0.))
        self.assertLess(reward(.1,.5),reward(.1,0.))
        self.assertEqual(reward(1.,1.,arr=True),0.)
        self.assertEqual(reward(1.,1.,lifted=False),0.)
        self.assertAlmostEqual(reward(math.radians(359),0.),reward(math.radians(-1),0.),places=5)

    def test_sampling_mixture_preserves_translation_and_nonzero_yaw_ranges(self):
        cfg=dict(self.cfg,in_place_probability=.25)
        torch.manual_seed(4);n=10000
        target=torch.full((n,2),4.);obj=torch.zeros(n,2);initial=torch.ones(n)*.1;old=initial+.2
        xy,yaw,mask=sample_in_place_targets(target,obj,initial,old,cfg)
        self.assertTrue(.23<mask.float().mean().item()<.27)
        torch.testing.assert_close(xy[mask],obj[mask]);torch.testing.assert_close(xy[~mask],target[~mask])
        torch.testing.assert_close(yaw[~mask],old[~mask])
        offset=yaw[mask]-initial[mask]
        self.assertGreaterEqual(offset.abs().min().item(),math.radians(10)-1e-6)
        self.assertLessEqual(offset.abs().max().item(),math.radians(30)+1e-6)
        self.assertTrue((offset<0).any());self.assertTrue((offset>0).any())
        state=torch.random.get_rng_state()
        sample_in_place_targets(target,obj,initial,old,self.cfg)
        self.assertTrue(torch.equal(state,torch.random.get_rng_state()))

    def test_new_configs_disable_alignment_and_keep_checkpoint_architecture(self):
        for stem in ['teamhoi_ltable_yaw']:
            old=yaml.safe_load((ROOT/'cfg'/(stem+'.yaml')).read_text())['params']
            new=yaml.safe_load((ROOT/'cfg'/(stem+'_transport.yaml')).read_text())['params']
            self.assertEqual(old['algo'],new['algo']);self.assertEqual(old['network'],new['network']);self.assertEqual(old['model'],new['model'])
            self.assertEqual(new['config']['asset_weights'],[1,1,1,3])
            self.assertEqual(new['config']['w'][8],0.)
            self.assertEqual(old['config']['w'][:8],new['config']['w'][:8])
            self.assertEqual(old['config']['w'][9],new['config']['w'][9])
            cfg=yaw_settings(new['config']['yaw'])
            self.assertTrue(cfg['staged_transport']);self.assertEqual(cfg['reference_loss_coef'],0.)

    def test_invalid_sampling_and_weights(self):
        for values in [dict(in_place_probability=1.1),dict(in_place_probability=.5),
                       dict(staged_transport=True,in_place_probability=.5,in_place_min_yaw_deg=2),
                       dict(carry_heading_weight=float('nan')),dict(carry_heading_scale_deg=0)]:
            with self.assertRaises(ValueError):yaw_settings(dict(enabled=True,**values))

    def test_subset_reset_rearms_pickup_without_touching_other_episodes(self):
        import numpy as np
        method=task_method('_reset_target',np=np,quat_from_angle_axis=lambda a,axis:yaw_quaternion(a))
        t=types.SimpleNamespace(device='cpu',_target_pos=torch.zeros(3,3),_target_end_pos=torch.zeros(3,3),
            _target_rot=yaw_quaternion(torch.zeros(3)),_target_dist_min=3.,_target_dist_max=10.,
            putdown_mask=torch.ones(3,dtype=torch.bool),arrived_mask=torch.ones(3,dtype=torch.bool),
            lifted_mask=torch.ones(3,dtype=torch.bool),in_place_mask=torch.ones(3,dtype=torch.bool),
            settled_count=torch.ones(3,dtype=torch.long)*7)
        method(t,torch.tensor([1]))
        for key in ['putdown_mask','arrived_mask','lifted_mask','in_place_mask']:
            self.assertEqual(getattr(t,key).tolist(),[True,False,True])
        self.assertEqual(t.settled_count.tolist(),[7,0,7])

    def test_subset_observations_gate_pickup_and_expose_phase_heading(self):
        def compute_obs(roots,pos,rim,vec,target,hand_z,putdown,num_humanoids):
            return roots.new_zeros((roots.shape[0],204+9*num_humanoids))
        method=task_method('_compute_task_obs_tensor',compute_obs=compute_obs)
        t=types.SimpleNamespace(device='cpu',num_envs=3,num_agents=2,yaw_enabled=True,yaw_config=self.cfg,
            viz=False,log_metrics=False,obj_height=.82,mask=torch.ones(3,2,dtype=torch.bool),
            _humanoid_root_states_tensor=torch.zeros(3,2,13),_obj_states=torch.zeros(3,13),
            _target_rot=yaw_quaternion(torch.ones(3)*.5),_target_end_pos=torch.zeros(3,3),
            _rigid_body_pos_tensor=torch.zeros(3,2,2,3),_lift_body_ids=torch.tensor([0,1]),
            obj_held_points=torch.zeros(3,64,3),task_obs_dim=[3,192,6,7,18],
            putdown_mask=torch.zeros(3,dtype=torch.bool),arrived_mask=torch.zeros(3,dtype=torch.bool),
            lifted_mask=torch.zeros(3,dtype=torch.bool),carry_yaw=torch.zeros(3),update_held_points=lambda *_:None)
        t._obj_states[:,6]=1;t._obj_states[:,2]=.85
        obs=method(t,torch.tensor([2]))
        self.assertFalse(t.arrived_mask.any());self.assertEqual(obs[0,0,207].item(),0.)
        t._obj_states[2,2]=.94;t._obj_states[2,3:7]=yaw_quaternion(torch.tensor(.1))
        obs=method(t,torch.tensor([2]))
        self.assertEqual(t.lifted_mask.tolist(),[False,False,True]);self.assertEqual(t.arrived_mask.tolist(),[False,False,True])
        self.assertAlmostEqual(t.carry_yaw[2].item(),.1,places=6)
        self.assertAlmostEqual(obs[0,0,204].item(),math.sin(.4),places=6)
        self.assertEqual(obs[0,0,207].item(),1.)


if __name__=='__main__':unittest.main()

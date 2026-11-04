"""Geometry-aware lifting and upper-height termination regression checks."""
import ast
import math
from pathlib import Path
import tempfile
import unittest
import torch
from test_yaw import ROOT
from utils.pose_control import axis_quaternion, quat_mul, pose_settings, pose_rotation_reward
from utils.table_clearance import load_collision_geometry, lowest_collision_offset, rotation_height_limit


class ClearanceTests(unittest.TestCase):
    def test_all_familiar_assets_match_original_upright_clearance(self):
        q=axis_quaternion(torch.tensor([0.]),0)
        for name in ('round','square','rectangle'):
            g=load_collision_geometry(ROOT/'assets'/('default_'+name+'.urdf'))
            bottom=lowest_collision_offset(q,g)
            torch.testing.assert_close(bottom,torch.tensor([-.84]))
            self.assertAlmostEqual((.97+bottom).item(),.13,places=6)
            self.assertAlmostEqual((.85+bottom).item(),.01,places=6)

    def test_rectangle_pitch_thirty_has_clearance_and_height_headroom(self):
        g=load_collision_geometry(ROOT/'assets/default_rectangle.urdf')
        q=axis_quaternion(torch.tensor([math.pi/6]),1)
        bottom=lowest_collision_offset(q,g)
        expected=-.84*math.cos(math.pi/6)-.90*math.sin(math.pi/6)
        self.assertAlmostEqual(bottom.item(),expected,places=6)
        required=.13-bottom
        self.assertGreater(required.item(),1.30)
        limit=rotation_height_limit(bottom,.13,.15)
        self.assertAlmostEqual(limit.item(),required.item()+.15,places=6)
        for yaw in (-2.,.7854,2.5):
            rotated=quat_mul(axis_quaternion(torch.tensor([yaw]),2),q)
            torch.testing.assert_close(lowest_collision_offset(rotated,g),bottom,atol=1e-6,rtol=1e-6)

    def test_sideways_and_upside_down_include_tabletop_not_only_legs(self):
        for name,side_extent in [('rectangle',1.),('round',1.),('square',.8)]:
            g=load_collision_geometry(ROOT/'assets'/('default_'+name+'.urdf'))
            angles=torch.tensor([math.pi/2,math.pi])
            bottom=lowest_collision_offset(axis_quaternion(angles,1),g)
            torch.testing.assert_close(bottom,torch.tensor([-side_extent,-.02]),atol=1e-6,rtol=1e-6)
        # At a half flip the tabletop can be low while the whole table is airborne.
        cfg=pose_settings({'enabled':True});clearance=torch.tensor([.15-.02])
        reward=pose_rotation_reward(torch.zeros(1,3),clearance+1e-5,torch.tensor([True]),cfg)
        self.assertEqual(reward.item(),1.)

    def test_rotated_collision_and_fixed_joint_origins(self):
        xml='''<robot name="test"><link name="root"/><link name="child"><collision>
        <origin xyz="0 0 .2" rpy="0 1.5707963267948966 0"/>
        <geometry><box size="2 .2 .4"/></geometry></collision></link>
        <joint name="fixed" type="fixed"><parent link="root"/><child link="child"/>
        <origin xyz="0 0 .5"/></joint></robot>'''
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'table.urdf';p.write_text(xml)
            g=load_collision_geometry(p)
        bottom=lowest_collision_offset(axis_quaternion(torch.tensor([0.]),0),g)
        self.assertAlmostEqual(bottom.item(),-.3,places=6)

    def test_cylinder_bound_matches_sampled_surface_and_caps(self):
        xml='''<robot name="test"><link name="root"><collision>
        <origin xyz=".1 0 .2" rpy=".2 .3 .4"/>
        <geometry><cylinder radius=".7" length=".1"/></geometry>
        </collision></link></robot>'''
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'table.urdf';p.write_text(xml);g=load_collision_geometry(p)
        q=quat_mul(axis_quaternion(torch.tensor([.8]),0),axis_quaternion(torch.tensor([.5]),1))
        # Independent dense sampling of the cylinder's two boundary circles.
        axis=g['axes'][0];u=torch.cross(axis,torch.tensor([0.,1.,0.]),dim=-1);u=u/u.norm();v=torch.cross(axis,u,dim=-1)
        theta=torch.linspace(0,2*math.pi,20000)
        circle=.7*(theta.cos()[:,None]*u+theta.sin()[:,None]*v)
        points=torch.cat([g['centers']+circle+.05*axis,g['centers']+circle-.05*axis])
        from utils.pose_control import rotate_inverse,quat_conjugate
        world=rotate_inverse(quat_conjugate(q).expand(points.shape[0],-1),points)
        self.assertAlmostEqual(lowest_collision_offset(q,g).item(),world[:,2].min().item(),places=5)

    def test_actual_reset_preserves_legacy_limit_and_accepts_valid_tilt(self):
        p=ROOT/'teamhoi/env/tasks/humanoid_multi_amp_teamhoi_task.py'
        fn=next(n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='compute_humanoid_reset')
        ns={'torch':torch};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(p),'exec'),ns)
        def reset(z,cap=None):
            return ns['compute_humanoid_reset'](torch.ones(4,dtype=torch.long),torch.ones(4,dtype=torch.long)*10,
                torch.zeros(4,15,3),torch.tensor([0]),torch.ones(4,15,3),900,True,
                torch.zeros(15),torch.tensor([[0.,0.,z]]),.70,4,obj_height_limit=cap)
        self.assertTrue(reset(1.31)[1].item())
        self.assertFalse(reset(1.31,torch.tensor([1.4575]))[1].item())
        self.assertTrue(reset(1.46,torch.tensor([1.4575]))[1].item())
        self.assertTrue(reset(.69,torch.tensor([1.4575]))[1].item())
        cap=rotation_height_limit(torch.tensor([-.84]),.13,.15)
        self.assertEqual(cap.item(),torch.tensor(1.30).item())

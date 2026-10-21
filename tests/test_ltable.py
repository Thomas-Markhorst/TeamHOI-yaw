"""L-table integration checks without Isaac Gym."""
import json
import math
from pathlib import Path
import sys
import unittest
import xml.etree.ElementTree as ET
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'teamhoi'))
from utils.polygon import asset_ranges, sample_rim
from utils.yaw_checkpoint import prepare_yaw_checkpoint
from test_yaw import task_method
import types


class LTableTests(unittest.TestCase):
    def test_weighted_mixture_and_default_allocation(self):
        ranges = asset_ranges(1024, 4, [1, 1, 1, 3])
        self.assertEqual([b-a for a,b in ranges], [171,171,170,512])
        self.assertEqual(asset_ranges(10,3), [(0,4),(4,7),(7,10)])
        for weights in ([0,1,1,1], [1,1], [math.nan]*4):
            with self.assertRaises(ValueError): asset_ranges(1024,4,weights)

    def test_mixed_asset_builder_keeps_observation_shape(self):
        cfgs = [json.loads((ROOT/'assets'/name).read_text()) for name in
                ('default_round.json','default_square.json','default_rectangle.json','l_table_com.json')]
        task = types.SimpleNamespace(asset_cfgs=cfgs,K_RIM=64,rim_points_list=[],obj_spans=[])
        method = task_method('_build_rim_points', sample_rim=sample_rim)
        method(task, device='cpu')
        self.assertEqual(len(task.rim_points_list),4)
        for points in task.rim_points_list:
            self.assertEqual(tuple(points.shape),(64,3))
            self.assertTrue(torch.isfinite(points).all())
        expected = torch.tensor(sample_rim(cfgs[-1]['rim_vertices'],64,cfgs[-1]['z_local']))
        torch.testing.assert_close(task.rim_points_list[-1],expected)

    def test_ltable_imported_horizontal_mass_reference(self):
        root = ET.parse(ROOT/'assets/l_table_com.urdf').getroot()
        offsets = {j.find('child').get('link'):list(map(float,j.find('origin').get('xyz').split()))
                   for j in root.findall('joint')}
        total = 0.; moment = [0.,0.]
        for link in root.findall('link'):
            inertial = link.find('inertial'); m=float(inertial.find('mass').get('value'))
            local=list(map(float,inertial.find('origin').get('xyz').split()))
            offset=offsets.get(link.get('name'),[0.,0.,0.]); total+=m
            for k in (0,1): moment[k]+=m*(local[k]+offset[k])
        self.assertAlmostEqual(total,65.088)
        for v in moment: self.assertAlmostEqual(v/total,0.,places=8)

    def test_yaw_warm_start_preserves_weights_and_rejects_bad_shapes(self):
        model={'a2c_network.task_encoder.1.0.weight':torch.randn(5,7),
               'a2c_network.task_encoder_critic.1.0.weight':torch.randn(5,7),'bias':torch.randn(5)}
        stats={'running_mean':torch.randn(431),'running_var':torch.ones(431),'count':torch.tensor(123.)}
        checkpoint={'model':model,'running_mean_std':stats,'yaw_reference':{'old_teacher':True}}
        copied, normalized=prepare_yaw_checkpoint(checkpoint,model,stats)
        for k in model:
            torch.testing.assert_close(copied[k],model[k]); self.assertNotEqual(copied[k].data_ptr(),model[k].data_ptr())
        for k in stats: torch.testing.assert_close(normalized[k],stats[k])
        bad=dict(model);bad['bias']=torch.zeros(6)
        with self.assertRaises(ValueError): prepare_yaw_checkpoint(checkpoint,bad,stats)
        with self.assertRaises(ValueError): prepare_yaw_checkpoint(checkpoint,model,{'running_mean':torch.zeros(427)})

    def test_experiment_only_changes_agreed_settings(self):
        base=yaml.safe_load((ROOT/'cfg/teamhoi_yaw.yaml').read_text())['params']['config']
        new=yaml.safe_load((ROOT/'cfg/teamhoi_ltable_yaw.yaml').read_text())['params']['config']
        self.assertEqual(base['w'],new['w'])
        for key in base['yaw']:
            if key != 'reference_loss_coef':self.assertEqual(base['yaw'][key],new['yaw'][key])
        self.assertEqual(new['yaw']['reference_loss_coef'],.5)
        self.assertEqual(new['asset_weights'],[1,1,1,3])

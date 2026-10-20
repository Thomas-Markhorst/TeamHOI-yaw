import ast
import math
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
import yaml
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'teamhoi'))
from utils.yaw_evaluation import evaluate_player
from utils.yaw import yaw_settings, placement_metrics, yaw_quaternion


class EvaluationTests(unittest.TestCase):
    def test_terminal_drift_cannot_count_as_success(self):
        task = types.SimpleNamespace(yaw_enabled=True, yaw_config=yaw_settings({'enabled':True}),
            mask=torch.ones(1,2,dtype=torch.bool), ranges=[(0,1)], assets=['table'],
            progress_buf=torch.zeros(1,dtype=torch.long),
            cfg={'args':types.SimpleNamespace(checkpoint='test.pth'),'seed':42})
        player = types.SimpleNamespace(env=types.SimpleNamespace(task=task))
        player.env_reset=lambda *args: {'obs':torch.zeros(1,2,1),'mask':task.mask}
        player.get_action=lambda *args,**kwargs: torch.zeros(2,28)
        def step(*args):
            task.progress_buf += 1
            final = task.progress_buf.item() == 2
            metrics={key:torch.tensor([value]) for key,value in dict(
                arrived=True,putdown=True,released=True,settled_success=not final,
                position_error=.01,yaw_error=.2 if final else 0.,angular_speed=0.).items()}
            return None,None,torch.tensor([final]),{'yaw_metrics':metrics,'terminate':torch.tensor([False])}
        player.env_step=step
        with tempfile.TemporaryDirectory() as folder:
            report=evaluate_player(player,1,str(Path(folder)/'result.json'))
        self.assertFalse(report['episodes'][0]['success'])
        self.assertAlmostEqual(report['episodes'][0]['yaw_error_deg'],math.degrees(.2),places=5)

    def test_hovering_or_toppled_table_is_not_placed(self):
        cfg=yaw_settings({'enabled':True, 'settled_steps':1})
        state=torch.zeros(1,13);state[:,6]=1;state[:,2]=.94
        target=torch.zeros(1,3);q=yaw_quaternion(torch.zeros(1))
        def success():
            return placement_metrics(state,target,q,torch.ones(1,2,2),torch.ones(1,2,dtype=torch.bool),
                torch.ones(1,dtype=torch.bool),torch.zeros(1,dtype=torch.long),cfg)[-1].item()
        self.assertFalse(success())
        state[:,2]=.82
        self.assertTrue(success())
        state[:,3:7]=torch.tensor([[1.,0.,0.,0.]])
        self.assertFalse(success())


class ConfigTests(unittest.TestCase):
    def setUp(self):
        source=ROOT/'teamhoi/utils/config.py'
        tree=ast.parse(source.read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='load_cfg')
        ns={'os':os,'yaml':yaml}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),ns)
        self.load=ns['load_cfg']

    def args(self, folder):
        return types.SimpleNamespace(cfg_train=str(ROOT/'cfg/teamhoi_yaw.yaml'),
            cfg_env=str(ROOT/'cfg/main.yaml'),num_envs=12,episode_length=900,
            task='HumanoidAMPTableLift',headless=True,system_max_humanoids=8,goal_multiplier=1.,
            min_humanoids=2,fix_num_humanoids=0,randomize=False,logdir='logs',torch_deterministic=False,
            exp_name='Yaw',experiment='Base',checkpoint='Base',llc_checkpoint='',max_iterations=0,
            seed=42,output_path=folder,init_from_checkpoint='',resume=0,test=False,play=False)

    def test_explicit_initialization_and_output_path_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            args=self.args(folder);args.init_from_checkpoint='original.pth'
            _,cfg,_=self.load(args)
            self.assertFalse(cfg['params']['load_checkpoint'])
            self.assertEqual(cfg['params']['config']['init_from_checkpoint'],'original.pth')
            path=Path(folder)/'Yaw/ckpt.pth';path.parent.mkdir();path.touch()
            with self.assertRaises(ValueError):self.load(args)
            args.init_from_checkpoint=''
            _,cfg,_=self.load(args)
            self.assertTrue(cfg['params']['load_checkpoint'])
            self.assertEqual(cfg['params']['load_path'],str(path))

    def test_initialization_cannot_be_used_as_resume_or_inference(self):
        with tempfile.TemporaryDirectory() as folder:
            args=self.args(folder);args.init_from_checkpoint='original.pth';args.test=True
            with self.assertRaises(ValueError):self.load(args)
            args.test=False;args.checkpoint='yaw.pth'
            with self.assertRaises(ValueError):self.load(args)

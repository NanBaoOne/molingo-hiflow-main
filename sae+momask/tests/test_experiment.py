import copy
import hashlib
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import types
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import numpy as np
import torch
from torch import nn

# Offline test double only; production always imports OpenAI CLIP.
class FakeClip(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1, 512))

    def encode_text(self, tokens):
        return self.weight.expand(tokens.shape[0], -1)


fake_clip = types.ModuleType('clip')
fake_clip.load = lambda *a, **kw: (FakeClip(), None)
fake_clip.tokenize = lambda texts, **kw: torch.ones(len(texts), 77, dtype=torch.long)
fake_clip.model = types.SimpleNamespace(convert_weights=lambda model: None)
sys.modules['clip'] = fake_clip

from components import SemanticRVQ, build_vq, build_transformer, pack_ema, restore_ema, local_alignment
from data272 import Motion272
import train

CONFIG = dict(mu=.99, num_quantizers=3, quantize_dropout_prob=0., nb_code=16, code_dim=16,
              width=16, depth=1, latent_dim=24, ff_size=32, layers=1, heads=2, dropout=0.)


def fixture(root):
    for part in ('motion_data', 'texts', 'babel_272_annotation_t5', 'mean_std', 'split'):
        (root / part).mkdir(parents=True)
    np.save(root / 'mean_std/Mean.npy', np.zeros(272, dtype=np.float32))
    np.save(root / 'mean_std/Std.npy', np.ones(272, dtype=np.float32))
    rng = np.random.default_rng(3)
    for i in range(4):
        name = f'{i:06}'
        np.save(root/'motion_data'/f'{name}.npy', rng.normal(size=(80,272)).astype(np.float32))
        # Time feature constant across channels -> easy crop alignment assertion.
        np.save(root/'babel_272_annotation_t5'/f'{name}.npy',
                np.broadcast_to(np.arange(80,dtype=np.float32)[:,None],(80,1024)).copy())
        (root/'texts'/f'{name}.txt').write_text('person walks#person/NOUN walks/VERB#0#0\n', encoding='utf-8')
    for split in ('train','val','test'):
        (root/'split'/f'{split}.txt').write_text('\n'.join(f'{i:06}' for i in range(4)))


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_upstream_files_unchanged(self):
        manifest = json.loads((HERE/'UPSTREAM.json').read_text())
        for path, expected in manifest['files'].items():
            self.assertEqual(hashlib.sha256((HERE/'vendor/momask'/path).read_bytes()).hexdigest(), expected)

    def test_file_logs_only_keep_metrics_and_validation_loss(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp)/'train.log'
            code = "print('ordinary setup'); print('10%|bar\\r20%|bar'); print('TRAIN_PROGRESS step=1'); print('VALIDATION_LOSS loss=0.1'); print('EVAL_METRICS fid=1.0')"
            result = subprocess.run([sys.executable,str(HERE/'log_stream.py'),'--file',str(log),'--',sys.executable,'-c',code],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            content = log.read_text()
            self.assertIn('VALIDATION_LOSS',content)
            self.assertIn('EVAL_METRICS',content)
            for unwanted in ('ordinary','TRAIN_PROGRESS','%|bar'):
                self.assertNotIn(unwanted,content)
                self.assertNotIn(unwanted,result.stdout)

    def test_log_rotation_is_bounded(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp)/'train.log'
            code = "for _ in range(500): print('EVAL_METRICS '+ 'x'*8192)"
            result = subprocess.run([sys.executable,str(HERE/'log_stream.py'),'--file',str(log),'--max-mb','1','--backups','2','--',sys.executable,'-c',code],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            files = list(Path(tmp).glob('train.log*'))
            self.assertEqual(len(files),3)
            self.assertTrue(all(p.stat().st_size <= 1024*1024 for p in files))

    def test_failures_go_to_separate_error_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp)/'train.log'
            result = subprocess.run([sys.executable,str(HERE/'log_stream.py'),'--file',str(log),'--',sys.executable,'-c',"raise RuntimeError('example failure')"],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertNotIn('example failure',log.read_text())
            self.assertIn('example failure',Path(str(log)+'.error.log').read_text())

    def test_semantic_and_native_rvq_backward(self):
        model = SemanticRVQ(CONFIG)
        x = torch.randn(2,64,272)
        y, commit, ppl, sem = model(x, torch.randn(2,64,1024), torch.tensor([64,60]), torch.tensor([True,False]))
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(torch.isfinite(ppl))
        loss = (y-x).square().mean() + commit*.02 + sem
        loss.backward()
        self.assertGreater(float(model.semantic_proj.weight.grad.abs().sum()), 0)
        self.assertGreater(float(next(model.vq.encoder.parameters()).grad.abs().sum()), 0)
        model.vq.eval()
        ids, _ = model.vq.encode(x)
        self.assertEqual(tuple(ids.shape), (2,16,3))
        self.assertEqual(model.vq.forward_decoder(ids).shape, x.shape)

    def test_semantic_masks_and_no_annotation(self):
        projection = nn.Linear(1024,16)
        z, t = torch.randn(2,16,16), torch.randn(2,64,1024)
        lens, has = torch.tensor([60,64]), torch.tensor([True,False])
        first = local_alignment(z,t,lens,has,projection)
        changed = t.clone()
        changed[1] = 999
        changed[0,60:] = 999
        z2 = z.clone()
        z2[0,15:] = 999
        second = local_alignment(z2,changed,lens,has,projection)
        self.assertTrue(torch.allclose(first,second))
        self.assertEqual(float(local_alignment(z,t,lens,torch.zeros_like(has),projection)),0.)

    def test_native_mask_residual_forward_and_generation(self):
        vq = build_vq(CONFIG)
        vq(torch.randn(2,64,272))  # initialize all EMA codebooks
        vq.eval()
        x = torch.randn(2,64,272)
        ids, _ = vq.encode(x)
        lengths, captions = torch.tensor([16,15]), ['walk','jump']
        mask = build_transformer(CONFIG,'mask','cpu')
        res = build_transformer(CONFIG,'res','cpu')
        for model, tokens in [(mask,ids[...,0]),(res,ids)]:
            loss, _, acc = model(tokens,captions,lengths)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
        base = mask.generate(captions,lengths,timesteps=2,cond_scale=2,gsample=True)
        full = res.generate(base,captions,lengths,num_res_layers=-1)
        self.assertEqual(full.shape,(2,16,3))
        self.assertTrue((full[1,15:] == -1).all())
        self.assertEqual(vq.forward_decoder(full).shape,(2,64,272))
        partial = res.generate(base,captions,lengths,num_res_layers=1)
        self.assertEqual(partial.shape,(2,16,2))
        self.assertTrue(torch.isfinite(vq.forward_decoder(partial)).all())

    def test_ema_resume_does_not_reset_codebooks(self):
        first = build_vq(CONFIG)
        first(torch.randn(2,64,272))
        second = build_vq(CONFIG)
        second.load_state_dict(copy.deepcopy(first.state_dict()))
        restore_ema(second,copy.deepcopy(pack_ema(first)),'cpu')
        x = torch.randn(2,64,272)
        torch.manual_seed(7)
        y1 = first(x)[0]
        torch.manual_seed(7)
        y2 = second(x)[0]
        self.assertTrue(torch.allclose(y1,y2))
        self.assertTrue(torch.equal(first.quantizer.codebooks,second.quantizer.codebooks))

    def test_data_semantics_share_crop_and_eval_is_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            ds = Motion272(root,'train',training=True,window=64,semantic=True)
            item = ds[0]
            offset = int(item['sample_id'].split(':')[1])
            self.assertEqual(item['babel'][0,0],offset)
            self.assertEqual(item['motion'].shape,(64,272))
            evaluation = Motion272(root,'test',training=False)
            self.assertTrue(torch.equal(evaluation[0]['motion'],evaluation[0]['motion']))
            # A malformed existing label must never silently become an unlabelled sample.
            np.save(root/'babel_272_annotation_t5/000000.npy',np.zeros((79,1024)))
            with self.assertRaises(ValueError):
                Motion272(root,'train',semantic=True)

    def test_training_save_and_resume_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            run = root/'experiment'
            evaluator = root/'evaluator.ckpt'
            evaluator.touch()
            cmd = ['train.py','--stage','vq','--device','cpu','--data-root',str(root),
                   '--evaluator',str(evaluator),'--run-dir',str(run),'--batch-size','2','--workers','0',
                   '--steps','2','--save-every','1','--eval-every','0','--val-batches','1',
                   '--code-dim','16','--nb-code','16','--num-quantizers','3','--width','16','--depth','1']
            # Real data/optimizer/EMA/save path; only real evaluator weights are unavailable locally.
            with patch('evaluation.evaluate',return_value={'fid':1.}):
                with patch.object(sys,'argv',cmd):
                    train.main()
                checkpoint = train.load_checkpoint(run/'vq/latest.pt')
                self.assertEqual(checkpoint['step'],2)
                self.assertTrue(all(e['init'] for e in checkpoint['ema']))
                cmd[cmd.index('--steps')+1] = '3'
                with patch.object(sys,'argv',cmd+['--resume']):
                    train.main()
                self.assertEqual(train.load_checkpoint(run/'vq/latest.pt')['step'],3)

    def test_mask_and_residual_training_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            run = root/'experiment'
            evaluator = root/'evaluator.ckpt'
            evaluator.touch()
            vq = build_vq(CONFIG)
            vq(torch.randn(2,64,272))
            train.atomic_save(dict(vq_model=vq.state_dict(),config=CONFIG,args={'lsa_weight':.001}),run/'vq/best_fid.pt')
            common = ['--device','cpu','--data-root',str(root),'--evaluator',str(evaluator),'--run-dir',str(run),
                      '--batch-size','2','--workers','0','--steps','2','--save-every','2',
                      '--eval-every','0','--val-batches','1','--latent-dim','24','--ff-size','32',
                      '--layers','1','--heads','2','--dropout','0']
            with patch('evaluation.evaluate',return_value={'fid':1.}):
                for stage in ('mask','res'):
                    with patch.object(sys,'argv',['train.py','--stage',stage]+common):
                        train.main()
                    ck = train.load_checkpoint(run/stage/'latest.pt')
                    self.assertEqual(ck['step'],2)
                    self.assertIsNotNone(ck['vq_sha256'])
                    self.assertFalse(any(k.startswith('clip_model.') for k in ck['model']))
                bad = train.load_checkpoint(run/'vq/best_fid.pt')
                # Even a same-path tokenizer replacement must be caught on resume.
                bad['vq_model']['encoder.model.0.weight'].add_(1)
                train.atomic_save(bad,run/'vq/best_fid.pt')
                resume = common.copy()
                resume[resume.index('--steps')+1] = '3'
                with patch.object(sys,'argv',['train.py','--stage','mask']+resume+['--resume']):
                    with self.assertRaisesRegex(ValueError,'contents changed'):
                        train.main()


if __name__ == '__main__':
    unittest.main()

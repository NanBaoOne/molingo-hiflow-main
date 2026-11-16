"""272-d / 30fps HumanML3D segments; shared crop for motion and BABEL."""
import json
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class Motion272(Dataset):
    def __init__(self, root, split, *, training=False, window=0, semantic=False,
                 max_length=300, min_length=60, babel_dir=None):
        self.root = Path(root)
        self.training, self.window, self.semantic = training, window, semantic
        self.babel_dir = Path(babel_dir) if babel_dir else self.root / 'babel_272_annotation_t5'
        self.max_length, self.unit = max_length, 4
        self.mean = np.load(self.root / 'mean_std/Mean.npy').astype(np.float32)
        self.std = np.load(self.root / 'mean_std/Std.npy').astype(np.float32)
        if self.mean.shape != (272,) or self.std.shape != (272,):
            raise ValueError('Expected 272-d normalization statistics')
        if not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or (self.std <= 0).any():
            raise ValueError('Invalid normalization statistics')
        self.records = []
        self.report = dict(split=split, missing=[], filtered=0, annotated=0)
        ids = (self.root / 'split' / f'{split}.txt').read_text().splitlines()
        for name in tqdm(ids, desc=f'Index {split}', dynamic_ncols=True):
            name = name.strip()
            if not name:
                continue
            mp, tp = self.root / 'motion_data' / f'{name}.npy', self.root / 'texts' / f'{name}.txt'
            if not mp.is_file() or not tp.is_file():
                self.report['missing'].append(name)
                continue
            shape = np.load(mp, mmap_mode='r').shape
            if len(shape) != 2 or shape[1] != 272:
                raise ValueError(f'{mp}: expected [T,272], got {shape}')
            n = shape[0]
            # Same source-length eligibility as existing molingo-hiflow loader.
            if not min_length <= n <= max_length:
                self.report['filtered'] += 1
                continue
            bp = self.babel_dir / f'{name}.npy'
            has = bp.is_file() if semantic else False
            if has:
                bs = np.load(bp, mmap_mode='r').shape
                if bs != (n, 1024):
                    raise ValueError(f'{bp}: expected {(n, 1024)}, got {bs}; no silent time resampling')
            full_captions = []
            for line in tp.read_text(encoding='utf-8').splitlines():
                fields = line.split('#')
                if len(fields) < 4:
                    raise ValueError(f'Malformed caption in {tp}: {line!r}')
                caption = fields[0]
                start, end = float(fields[2]), float(fields[3])
                start = 0. if np.isnan(start) else start
                end = 0. if np.isnan(end) else end
                if start == 0 and end == 0:
                    full_captions.append(caption)
                else:
                    a, b = max(0, int(start * 30)), min(n, int(end * 30))
                    if min_length <= b - a <= max_length:
                        self.records.append((name, a, b, [caption], has))
            if full_captions:
                self.records.append((name, 0, n, full_captions, has))
        self.report['records'] = len(self.records)
        self.report['annotated'] = sum(r[4] for r in self.records)
        print(json.dumps(self.report, ensure_ascii=False), flush=True)
        if not self.records:
            raise ValueError(f'No eligible samples in {root}/{split}')
        if semantic and self.report['annotated'] == 0:
            raise ValueError(f'LSA enabled but no BABEL features found in {self.babel_dir}')
        if self.report['missing']:
            raise FileNotFoundError(f'Missing motion/text files for {len(self.report["missing"])} IDs; first: {self.report["missing"][:5]}')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        name, a, b, captions, has = self.records[index]
        length = (b - a) // self.unit * self.unit
        if self.window:
            length = min(length, self.window)
        elif self.training and length >= 8 and random.random() < 1 / 3:
            length -= self.unit
        offset = random.randint(0, b - a - length) if self.training else 0
        a += offset
        m = np.array(np.load(self.root / 'motion_data' / f'{name}.npy', mmap_mode='r')[a:a+length], dtype=np.float32)
        m = (m - self.mean) / self.std
        target = self.window or self.max_length
        m = np.pad(m, ((0, target - length), (0, 0)))
        item = dict(motion=torch.from_numpy(m), length=length,
                    caption=random.choice(captions) if self.training else captions[0],
                    sample_id=f'{name}:{a}:{length}', has_babel=has)
        if self.semantic:
            t = np.zeros((target, 1024), dtype=np.float32)
            if has:
                t[:length] = np.load(self.babel_dir / f'{name}.npy', mmap_mode='r')[a:a+length]
            item['babel'] = torch.from_numpy(t)
        return item

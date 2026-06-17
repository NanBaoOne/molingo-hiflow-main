#!/usr/bin/env python3
"""Disable periodic latest checkpoint saves while keeping best checkpoints.

Run from either:
  - molingo-hiflow: python patch_save_best_only.py
  - mogen-hiflow:   python ../patch_save_best_only.py
"""

from __future__ import annotations

import py_compile
import shutil
from pathlib import Path


TRAINER_ORIGINAL = '''            if epoch >= 1 and epoch % self.opt.save_every_e == 0:
                misc.save_model(self.opt.model_dir, model_without_ddp=model_without_ddp, optimizer=self.optimizer,
                                loss_scaler=loss_scaler, epoch=epoch, ema_params=ema_params, epoch_name="latest")
'''

TRAINER_PATCHED = '''            # Periodic latest checkpoints are intentionally disabled; only metric-best models are saved.
'''

SAVE_EVERY_ORIGINAL = """    parser.add_argument('--save_every_e', type=int, default=150, help='save every this many epochs')"""

SAVE_EVERY_PATCHED = """    parser.add_argument('--save_every_e', type=int, default=150,
                        help='kept for compatibility; MoLingo no longer saves periodic latest checkpoints')"""


def find_mogen_root() -> Path:
    script_dir = Path(__file__).resolve().parent
    cwd = Path.cwd().resolve()
    candidates = [
        cwd,
        cwd / "mogen-hiflow",
        cwd.parent / "mogen-hiflow",
        script_dir / "mogen-hiflow",
        script_dir.parent / "mogen-hiflow",
    ]

    for candidate in candidates:
        trainer = candidate / "core" / "molingo_trainer.py"
        option = candidate / "options" / "molingo_option.py"
        if trainer.is_file() and option.is_file():
            return candidate

    checked = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError("Could not locate mogen-hiflow with required files. Checked:\n" + checked)


def patch_trainer(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    if TRAINER_ORIGINAL not in text:
        if TRAINER_PATCHED in text:
            return False
        raise RuntimeError("Periodic latest checkpoint block was not found in core/molingo_trainer.py")

    path.write_text(text.replace(TRAINER_ORIGINAL, TRAINER_PATCHED, 1), encoding="utf-8")
    return True


def patch_option(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    if SAVE_EVERY_PATCHED in text:
        return False
    if SAVE_EVERY_ORIGINAL not in text:
        return False

    path.write_text(text.replace(SAVE_EVERY_ORIGINAL, SAVE_EVERY_PATCHED, 1), encoding="utf-8")
    return True


def compile_and_clean(paths: list[Path]) -> None:
    for path in paths:
        py_compile.compile(str(path), doraise=True)

    for path in paths:
        cache_dir = path.parent / "__pycache__"
        if cache_dir.exists():
            shutil.rmtree(cache_dir)


def main() -> None:
    root = find_mogen_root()
    trainer = root / "core" / "molingo_trainer.py"
    option = root / "options" / "molingo_option.py"

    trainer_changed = patch_trainer(trainer)
    option_changed = patch_option(option)
    compile_and_clean([trainer, option])

    print(f"mogen-hiflow root: {root}")
    print(f"trainer patched: {trainer_changed}")
    print(f"option help patched: {option_changed}")
    print("py_compile passed and __pycache__ cleaned")


if __name__ == "__main__":
    main()

"""Import untouched MoMask modules and the sibling evaluator without symlinks."""
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE / 'vendor' / 'momask'))
# mogen-hiflow is not a valid Python package name. Expose its existing source
# under its expected import name; do not rename/mutate the server installation.
if 'mogen' not in sys.modules:
    package = types.ModuleType('mogen')
    package.__path__ = [str(ROOT / 'mogen-hiflow')]
    sys.modules['mogen'] = package

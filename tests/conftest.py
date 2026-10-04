"""
pytest 共通設定。

* リポジトリ直下を import パスの先頭に入れる (cuda_graph_step.py / data_utils.py / losses.py などを import するため)。
* losses.py が import する重い依存 (torchaudio / transformers) と、CUDA 専用の MAS カーネル (super_monotonic_align)
  が環境に無ければ、空のスタブに差し替える。これらは CPU / meta のテストでは使わない。
  (実物が入っている環境では何もしない)
"""

import importlib.util
import os
import sys
import types

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

if importlib.util.find_spec("torchaudio") is None:
    sys.modules.setdefault("torchaudio", types.ModuleType("torchaudio"))
if importlib.util.find_spec("transformers") is None:
    _tr = types.ModuleType("transformers")
    _tr.AutoModel = object  # type: ignore[attr-defined]
    sys.modules.setdefault("transformers", _tr)
if importlib.util.find_spec("super_monotonic_align") is None:
    _sma = types.ModuleType("super_monotonic_align")
    _sma.maximum_path = lambda *a, **k: None  # type: ignore[attr-defined]
    sys.modules.setdefault("super_monotonic_align", _sma)

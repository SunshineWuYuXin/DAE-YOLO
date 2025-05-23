'''
    @from MangoAI &3836712GKcH2717GhcK. please see https://github.com/iscyy/ultralyticsPro
'''
from ultralytics.utils import IterableSimpleNamespace
from pathlib import Path
import sys
from ultralytics.utils import DEFAULT_CFG
from ultralytics.cfg import get_cfg
import platform
current_os = platform.system()
import yaml
import re

def yaml_load(file="data.yaml", append_filename=False):

    assert Path(file).suffix in (".yaml", ".yml"), f"Attempting to load non-YAML file {file} with yaml_load()"
    with open(file, errors="ignore", encoding="utf-8") as f:
        s = f.read()  # string

        if not s.isprintable():
            s = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+", "", s)

        data = yaml.safe_load(s) or {}  
        if append_filename:
            data["yaml_file"] = str(file)
        return data

def load_script():
    original_path = sys.argv[2]
    new_path = original_path.replace("ultralytics\\", "").replace("\\", "/")

    FILE = Path(__file__).resolve()
    if current_os == "Windows":
        ROOT = FILE.parents[1]
    elif current_os == "Linux":
        ROOT = FILE.parents[2]
    else:
        ROOT = FILE.parents[2]
    DEFAULT_CFG_PATH = ROOT / new_path
    DEFAULT_CFG_DICT = yaml_load(DEFAULT_CFG_PATH)
    DEFAULT_CFG_PA = IterableSimpleNamespace(**DEFAULT_CFG_DICT)
    return DEFAULT_CFG_PA

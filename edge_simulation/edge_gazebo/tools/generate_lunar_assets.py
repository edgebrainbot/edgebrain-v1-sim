"""Regenerate the textured lunar assets from the original project meshes."""
from pathlib import Path
import runpy

if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).with_name('refine_lunar_assets.py')), run_name='__main__')


# standard imports
from os.path import abspath, join, dirname, pardir
import sys

# add local scripts path to PYTHON PATH
scripts_dirpath = abspath(join(dirname(__file__), pardir, "scripts"))
sys.path.append(scripts_dirpath)

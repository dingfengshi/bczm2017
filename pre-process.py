import config
from loaddata import loaddata

option = config.getoption()

load = loaddata()
load.pro_traingset(has_flag=True)


#-*- coding:utf-8 -*-

import config
from loaddata import loaddata

option=config.getoption()

load=loaddata()
load.create_index_vocab()
load.pro_traingset()

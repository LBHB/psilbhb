
import os
import shutil

from psilbhb.util.celldb import celldb, flush_training, readpsievents

c = celldb()

if os.path.exists("d:/data/Reishi"):
    flush_training("REI", c=c, local_folder="d:/data")


if os.path.exists("e:/data/SpindleShank"):
    #flush_training("REI", c=c, local_folder="d:/data")
    flush_training("SDS", c=c, local_folder="e:/data")
    flush_training("CGL", c=c, local_folder="e:/data")
    flush_training("IKI", c=c, local_folder="e:/data")




import os
import shutil

from psilbhb.util.celldb import celldb, flush_training, readpsievents

c = celldb()
#flush_training("LMD",c=c)
#flush_training("SQD", c=c, local_folder="d:/data")
flush_training("SDS", c=c, local_folder="e:/data")

#flush_training("REI", c=c, local_folder="d:/data")

flush_training("CGL", c=c, local_folder="e:/data")
flush_training("IKI", c=c, local_folder="e:/data")



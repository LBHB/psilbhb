
import os
import shutil
import socket

from psilbhb.util.celldb import celldb, flush_training, readpsievents

hostname = socket.gethostname()

c = celldb()

if hostname == 'badger':
    #flush_training("SQD", c=c, local_folder="d:/data")
    flush_training("REI", c=c, local_folder="e:/data")
    flush_training("SDS", c=c, local_folder="e:/data")
    flush_training("DRY", c=c, local_folder="e:/data")
    flush_training("IKI", c=c, local_folder="e:/data")

elif hostname == 'weasel':

    # flush_training("REI", c=c, local_folder="d:/data")
    flush_training("SQD", c=c, local_folder="d:/data")
    flush_training("SDS", c=c, local_folder="d:/data")
    flush_training("IKI", c=c, local_folder="d:/data")
else:
    raise ValueError(f"Unknown hostname {hostname}")


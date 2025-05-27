
import os
import shutil
import socket

from psilbhb.util.celldb import celldb, flush_training, readpsievents

hostname = socket.gethostname()

c = celldb()
c.user='david'
rawid = 147511

draw = c.pd_query(f"SELECT * FROM gDataRaw WHERE id={rawid}")
psipath = draw.loc[0,'resppath'] + draw.loc[0,'parmfile']
runclass = draw.loc[0,'runclass']
d, dataparm, dataperf = readpsievents(psipath, runclass)
dataparm['audio'] = 'Free-field'
dataparm['io'] = 'psilbhb.config.psi.io.tateril'

c.sqlupdate('gDataRaw', rawid, d=d, idfield='id')
c.save_data(rawid, dataparm, parmtype=0, keep_existing=False)
c.save_data(rawid, dataperf, parmtype=1, keep_existing=False)


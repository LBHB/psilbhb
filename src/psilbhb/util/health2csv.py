
import os
import shutil

from psilbhb.util.celldb import celldb

c = celldb()
sql="SELECT animal, date, weight, water, cleaned, surgery, special_treatment, notes FROM gHealth WHERE animal<>'Test' AND animal<>'Python' AND date>=date_sub(now(), interval 1 month) ORDER BY animal, date"
df = c.pd_query(sql)
df.to_csv('/tmp/health.csv')
df.to_html('/tmp/health.html')

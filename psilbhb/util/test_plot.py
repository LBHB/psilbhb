
import matplotlib
matplotlib.use('QtAgg')
from psilbhb.util import plots
from psilbhb.util.celldb import celldb, readpsievents, readlogs
from psilbhb.util.plots import plot_behavior, fix_old_plots
from psi import get_config

import matplotlib.pyplot as plt

plt.ion()

rawid = 150067
#rawid = 149974
#df_trial = plot_behavior(rawid=rawid, save_fig=True)
#df_trial=plot_behavior(150093, save_fig=True)
#dbias = d.groupby['snr',

#d.groupby('response')['correct'].count()

fix_old_plots("SQD07")
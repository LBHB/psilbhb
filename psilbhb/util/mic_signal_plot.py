import pandas as pd
from psiaudio import util
import matplotlib.pyplot as plt
plt.ion()

#filename = r'D:\data\Test\training2024\Test_2024_12_17_BLT_29'
#filename = r'D:\data\Test\training2024\Test_2024_12_26_FTC_19'
#filename = r'D:\data\Test\training2024\Test_2024_12_26_FTC_60'
#filename = r'D:\data\Test\training2025\Test_2025_01_07_FTC_11'
filename = r'D:\data\Test\training2025\Test_2025_01_07_BNT_13'
#dpgram_cal = r'D:\cfts\20241226-112014 test test left test dpgram\dpoae_probe_calibration.csv'

#df = pd.read_csv(dpgram_cal)
#plt.plot(df['frequency'], df['norm_spl'], 'k-')

#df = pd.read_csv(filename + f'\starship_1_calibration.csv')
#plt.plot(df['frequency'], df['norm_spl'], 'r-')

#plt.show()
#import sys
#sys.exit()

from psidata.api import Recording

fh = Recording(filename)

t0 = fh.trial_log['trial_start'].values
duration = fh.trial_log['duration'].max()


#mic_name = 'microphone_calibration'
mic_name = 'microphone_2'
mic = getattr(fh, mic_name)

cal = mic.get_calibration()
print(cal.sensitivity)
epochs = mic.get_segments(t0, offset=1, duration=4)

plt.figure()
plt.plot(epochs.iloc[0])
plt.plot(epochs.iloc[1])
plt.plot(epochs.iloc[2])
plt.plot(epochs.iloc[3])
#plt.show()
#import sys
#sys.exit()

from scipy.signal import periodogram
plt.close('all')
f,ax=plt.subplots()
for i,r in epochs.iterrows():
    f, Pxx_den = periodogram(r.values, mic.fs, nfft=512, detrend='linear')
    ax.plot(f, Pxx_den)
ax.set_xscale('log')
ax.set_ylim([-5e-13,5e-13])


psd = util.psd_df(epochs, fs=mic.fs)
spl = cal.get_db(psd)

plt.figure()
plt.axhline(80)
plt.plot(spl.T, color='0.5')
#plt.plot(cal.frequency, cal.sensitivity - 80, 'k-')
plt.xscale('log')
plt.axis(xmin=200, xmax=20e3, ymax=100, ymin=50)
#plt.show()
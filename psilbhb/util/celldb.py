import logging

import datetime
import json
import numpy as np
import os
import shutil

from sqlalchemy import create_engine, desc, exc
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.sql import text

import pandas as pd
import pandas.io.sql as psql

#import settings
from psi import get_config
from psi.util import PSIJsonEncoder

log = logging.getLogger(__name__)

def readlogs(logpath=None, rawid=None, c=None):
    if c is None:
        c=celldb()
    if logpath is None:
        rawdata = c.pd_query(f"SELECT * FROM gDataRaw where id={rawid}")
        logpath=rawdata.loc[0,'resppath'] + rawdata.loc[0,'parmfile']
        if not(os.path.isdir(logpath)):
            logpath = logpath.replace('d:', 'e:')
        if not(os.path.isdir(logpath)):
            logpath = logpath.replace('/auto/data/daq', 'h:/daq')
    eventlogfile = os.path.join(logpath, 'event_log.csv')
    triallogfile = os.path.join(logpath, 'trial_log.csv')
    df_trial = pd.read_csv(triallogfile)
    df_event = pd.read_csv(eventlogfile)
    return df_trial, df_event


def readpsievents(logpath=None, runclass=None, rawid=None, c=None):
    """

    Parameters
    ----------
    logpath
    runclass
    rawid
    c: celldb instance

    Returns
    -------

    """
    # make sure correct path separator is used.
    if logpath is None:
        d = c.pd_query(f"SELECT * FROM gDataRaw where id={rawid}")
        logpath = os.path.join(d['resppath'].iloc[0],d['parmfile'].iloc[0])
        runclass = d['runclass'].iloc[0]

    logpath = logpath.replace("/", os.path.sep)
    logpath = logpath.replace("\\", os.path.sep)
    eventlogfile = os.path.join(logpath, 'event_log.csv')
    triallogfile = os.path.join(logpath, 'trial_log.csv')
    print('triallogfile=', triallogfile)
    df = pd.read_csv(triallogfile)
    rawdata = {'trials': df.shape[0]}
    if 'correct' in df.columns:
        rawdata['corrtrials'] = df.correct.sum()
    else:
        rawdata['corrtrials'] = 0
    if 'current_full_rep' in df.columns:
        rawdata['reps'] = df['current_full_rep'].max()
    else:
        rawdata['reps'] = 1
    if runclass is None:
        runclass = logpath[-3:]
    if runclass=='NTD':
        row = df.iloc[-1]
        parmnames = ['target_delay','target_tone_start_time','target_tone_rise_time',
                     'target_tone_frequency','target_tone_polarity','target_tone_phase',
                     'hold_duration','response_duration',
                     'background_wav_sequence_path','background_wav_sequence_level','background_wav_sequence_duration',
                     'background_wav_sequence_normalization','background_wav_sequence_norm_fixed_scale',
                     'background_wav_sequence_fit_range', 'background_wav_sequence_fit_reps',
                     'background_wav_sequence_test_range','background_wav_sequence_test_reps',
                     'background_wav_sequence_channel_config','background_wav_sequence_random_seed',
                     'background_1_wav_sequence_path','background_1_wav_sequence_level','background_1_wav_sequence_duration',
                     'background_1_wav_sequence_normalization','background_1_wav_sequence_norm_fixed_scale',
                     'background_1_wav_sequence_fit_range','background_1_wav_sequence_fit_reps',
                     'background_1_wav_sequence_test_range','background_1_wav_sequence_test_reps',
                     'background_1_wav_sequence_channel_config','background_1_wav_sequence_random_seed',
                     'iti_duration', 'to_duration',
                     'water_dispense_duration', 'go_probability', 'repeat_fa',
                     'remind_trials', 'warmup_trials', 'min_nogo', 'max_nogo'
                     ]
        dataparm = {k: row[k] for k in parmnames if k in row.index}

        sdtfile = os.path.join(logpath, 'sdt_analysis.csv')
        if os.path.exists(sdtfile):
            df_perf = pd.read_csv(sdtfile);

            perfname =['trial_type', 'snr', 'n_correct', 'n_trials', 'fraction_correct',
                       'z_score', 'reference_z_score']
            dataperf = {k: list(df_perf[k]) for k in perfname}
        else:
            dataperf = {}
    elif runclass in ['NFB', 'CAT']:
        parmnames = ['fg_duration',
         'bg_duration', 'fg_snr', 'fg_delay', 'fg_channel', 'bg_channel',
         'response_condition', 'response_window', 'current_full_rep',
         'primary_channel',
         'combinations', 'fg_switch_channels', 'bg_switch_channels',
         'fg_go_index', 'random_seed', 'fg_path', 'fg_level', 'fg_normalization',
         'fg_norm_fixed_scale', 'fg_fit_range', 'fg_fit_reps', 'fg_test_range',
         'fg_test_reps', 'fg_channel_count', 'fg_binaural_combinations',
         'fg_channel_offset', 'bg_path', 'bg_level', 'bg_normalization',
         'bg_norm_fixed_scale', 'bg_fit_range', 'bg_fit_reps', 'bg_test_range',
         'bg_test_reps', 'bg_channel_count', 'bg_binaural_combinations',
         'bg_channel_offset', 'repeat_incorrect', 'iti_duration', 'to_duration',
         'response_duration', 'target_delay', 'np_duration', 'hold_duration',
         'training_mode', 'manual_control', 'keep_lights_on',
         'water_dispense_duration']
        if runclass == 'CAT':
            parmnames.extend(['bgc_path', 'bgc_level', 'bgc_normalization',
            'bgc_norm_fixed_scale', 'bgc_fit_range', 'bgc_fit_reps', 'bgc_test_range', 'bgc_test_reps'])
        row = df.iloc[-1]
        dataparm = {k: row[k] for k in parmnames if k in row.index}

        if 'score' in df.columns:
            correct_trials = (df['score'] == 2)

            dataperf = {'trials': df.shape[0],
                        'correct': (df['score']==2).sum(),
                        'invalid': (df['score']==0).sum(),
                        'incorrect': (df['score']==1).sum(),
                        'repeat_trials': df['trial_is_repeat'].sum(),
                        'rt': (df.loc[correct_trials,'response_ts']-
                               df.loc[correct_trials,'response_start']).mean()
                        }
        else:
            dataperf = {}

    elif runclass in ['VOW', 'VGN']:
        parmnames = ['sound_path', 'target_set', 'non_target_set', 'catch_set',
                     'switch_channels', 'repeat_count', 'repeat_isi', 'tar_to_cat_ratio',
                     'level', 'fs', 'response_end', 'random_seed', 'repeat_incorrect', 'snr',
                     'iti_duration', 'to_duration', 'response_duration', 'target_delay',
                     'np_duration', 'hold_duration', 'training_mode', 'manual_control',
                     'keep_lights_on', 'water_dispense_duration']
        row = df.iloc[-1]
        dataparm = {k: row[k] for k in parmnames if k in row.index}

        correct_trials = (df['score'] == 2)
        dataperf = {'trials': df.shape[0],
                    'correct': (df['score'] == 2).sum(),
                    'invalid': (df['score'] == 0).sum(),
                    'incorrect': (df['score'] == 1).sum(),
                    'repeat_trials': df['trial_is_repeat'].sum(),
                    'rt': (df.loc[correct_trials, 'response_ts'] -
                           df.loc[correct_trials, 'np_start']).mean()
                    }
    elif runclass in ['AMF']:
        parmnames = ['target_frequency', 'target_am_rate', 'target_bandwidth', 'modulation_depth',
                     'target_level', 'distractor_frequency', 'distractor_level', 'duration',
                     'primary_channel', 'switch_channels', 'reward_ambiguous_frac', 'fs', 'response_start', 'response_end',
                     'random_seed']
        row = df.iloc[-1]
        dataparm = {k: row[k] for k in parmnames if k in row.index}

        correct_trials = (df['score'] == 2)
        dataperf = {'trials': df.shape[0],
                    'correct': (df['score'] == 2).sum(),
                    'invalid': (df['score'] == 0).sum(),
                    'incorrect': (df['score'] == 1).sum(),
                    'repeat_trials': df['trial_is_repeat'].sum(),
                    'rt': (df.loc[correct_trials, 'response_ts'] -
                           df.loc[correct_trials, 'np_start']).mean()
                    }
    else:
        log.info(f"runclass {runclass}, assuming passive")
        parmnames = list(df.columns)
        parmnames = [c for c in parmnames if not c.startswith('this_')]
        parmnames = [c for c in parmnames if not c.startswith('trial_')]
        parmnames = [c for c in parmnames if not c in ['wav_set_idx','current_full_rep']]
        row = df.iloc[-1]
        dataparm = {k: row[k] for k in parmnames if k in row.index}
        dataperf = {}

    return rawdata, dataparm, dataperf


class celldb():

    ENGINE = None
    user = None
    animal = None
    training = 1
    MYSQL_USER = None
    MYSQL_PASS = None
    MYSQL_HOST = None
    MYSQL_PORT = 3306
    MYSQL_DB = None
    TESTMODE = False

    def __init__(self):
        self.MYSQL_USER = get_config('MYSQL_USER')
        self.MYSQL_PASS = get_config('MYSQL_PASS')
        self.MYSQL_HOST = get_config('MYSQL_HOST')
        self.MYSQL_PORT = get_config('MYSQL_PORT')
        self.MYSQL_DB = get_config('MYSQL_DB')
        self.TESTMODE = get_config('MYSQL_TESTMODE')

    def Engine(self):
        '''Returns a mysql engine object. Creates the engine if necessary.
        Otherwise returns the existing one.'''
        global __ENGINE__

        uri = self.get_db_uri()
        if self.ENGINE is None:
            self.ENGINE = create_engine(uri, pool_recycle=1600)

        return self.ENGINE

    def Session(self):
        '''Returns a mysql session object.'''
        engine = self.Engine()
        return sessionmaker(bind=engine)()

    def Tables(self):
        '''Returns a dictionary containing Narf database table objects.'''
        engine = self.Engine()
        Base = automap_base()
        Base.prepare(engine, reflect=True)

        tables = {
                'gAnimal': Base.classes.gAnimal,
                'gPenetration': Base.classes.gPenetration,
                'gCellMaster': Base.classes.gCellMaster,
                'gDataRaw': Base.classes.gDataRaw,
                'gData': Base.classes.gData,
                'gSingleCell': Base.classes.gSingleCell,
                'gSingleRaw': Base.classes.gSingleRaw,
                }

        return tables

    def get_db_uri(self):
        '''Used by Engine() to establish a connection to the database.'''

        db_uri = 'mysql+pymysql://{0}:{1}@{2}:{3}/{4}'.format(
                self.MYSQL_USER, self.MYSQL_PASS, self.MYSQL_HOST,
                self.MYSQL_PORT, self.MYSQL_DB
                )
        return db_uri

    def pd_query(self, sql=None, params=None):
        """
        execute an SQL command and return the results in a dataframe
        """

        if sql is None:
            raise ValueError("parameter sql required")
        engine = self.Engine()

        d = None
        try:
            d = pd.read_sql_query(sql=text(sql), con=engine, params=params)
        except exc.SQLAlchemyError as OpErr:
            if OpErr._message().count('Lost connection to MySQL server during query')>0:
                log.warning('Lost connection to MySQL server during query, trying again.')
                d = pd.read_sql_query(sql=sql, con=engine, params=params)

        if d is None:
            raise ValueError(f"pd_query error on sql={sql} params={params}")
        return d

    def sqlexec(self, sql):
        engine = self.Engine()
        conn = engine.connect()
        conn.execute(text(sql))

        return True

    def sqlinsert(self, table, d):
        """
        TODO: what if d is too long?
        """
        engine = self.Engine()
        conn = engine.connect()

        # creating column list for insertion
        cols = "`,`".join([str(i) for i in d.columns.tolist()])
        #sql = f"REPLACE INTO `{table}` (`" +cols + "`) VALUES (" + "%s,"*(len(d.columns.tolist())-1) + "%s)"
        sql = f"REPLACE INTO `{table}` (`" +cols + "`) VALUES "
        print(sql)
        values = []
        for i,row in d.iterrows():
            r=[]
            for _r in list(row.values):
                if type(_r) is bool:
                    r.append(str(int(_r)))
                elif type(_r) is str:
                    r.append('"'+_r.replace('"','')+'"')
                else:
                    r.append(str(_r))
            v = '(' + ','.join(r) + ')'
            print(v)

            values.append(v)
        sql = sql + ",".join(values)

        # Insert DataFrame in a bulk insert
        print(sql)

        if self.TESTMODE:
            print(sql)
            return sql
        else:
            res = conn.execute(text(sql))
            return res.lastrowid

    def sqlupdate(self, table, id, d=None, idfield='id'):
        """
        TODO: what if d is too long?
        """
        engine = self.Engine()
        conn = engine.connect()

        fv = []
        for k,v in d.items():
            if type(v) is str:
                fv.append(f"{k}='{v}'")
            else:
                fv.append(f"{k}={v}")

        if type(id) is str:
            id = "'" + id + "'"
        sql = f"UPDATE {table} SET {','.join(fv)} WHERE {idfield}={id}"

        if self.TESTMODE:
            print(sql)
        else:
            conn.execute(text(sql))

    def get_users(self):
        d = self.pd_query("SELECT id,userid,email FROM gUserPrefs WHERE active")
        return d

    def get_animals(self, name=None, lab="lbhb", species="ferret", active=True):
        if active:
            active_string = " AND onschedule<2"
        else:
            active_string = ""

        if name is None:
            sql = f"SELECT * FROM gAnimal WHERE lab='{lab}' {active_string} AND species='{species}'"
        else:
            sql = f"SELECT * FROM gAnimal WHERE animal='{name}'"
        d = self.pd_query(sql)

        return d

    def get_penetration(self, penname):
        sql = f"SELECT * FROM gPenetration WHERE penname='{penname}'"
        #print(sql)
        return self.pd_query(sql)

    def get_rawdata(self, siteid=None, rawid=None):
        if rawid is None:
            wherestr = f"WHERE gDataRaw.cellid='{siteid}'"
        else:
            wherestr = f"WHERE gDataRaw.id={rawid}"
        sql = "SELECT gDataRaw.*, gCellMaster.penid" +\
              " FROM gDataRaw INNER JOIN gCellMaster" +\
              f" ON gDataRaw.masterid=gCellMaster.id {wherestr} ORDER BY id"

        return self.pd_query(sql)

    def last_pen_data(self, animal=None, training=None):
        """
        get more recent entry in gPenetration for animal/training condition.
        :param animal:
        :param training:
        :return: pendata: dataframe
        """
        if animal is None:
            animal = self.animal
        if training is None:
            training = self.training

        sql = f"SELECT max(id) as maxid FROM gPenetration WHERE training={training} AND animal='{animal}'"
        lastpendata = self.pd_query(sql)
        maxid = lastpendata.loc[0, 'maxid']
        if maxid is None:
            maxid=0

        sql = f"SELECT * FROM gPenetration WHERE id={maxid}"
        lastpendata = self.pd_query(sql)

        return lastpendata

    def get_current_penname(self, animal=None, training=None,
                            force_next=False, create_if_missing=True):
        """
        get penname for animal/training condition. if latest pendate is before today,
        increment penname
        :param animal:
        :param training:
        :return: penname: str
        """
        if animal is None:
            animal = self.animal
        if training is None:
            training = self.training

        d_animal = self.get_animals(name=animal)
        if len(d_animal)>0:
            cellprefix = d_animal.loc[0, 'cellprefix']
        else:
            raise ValueError(f"Animal {animal} not found in celldb.gAnimal")


        today = datetime.date.today().strftime("%Y-%m-%d")
        need_to_create = False

        lastpendata = self.last_pen_data(animal=animal, training=training)
        if len(lastpendata) == 0:
            print('No penetrations exist for this animal. Guessing info from scratch.')
            pennum=1
        else:
            penname = lastpendata.loc[0,'penname']
            pendate = lastpendata.loc[0,'pendate']

            if (pendate == today) & ~force_next:
                pennum = int(penname[3:6])
            else:
                print(f"Last pendate: {pendate} incrementing from {penname}")
                pennum = int(penname[3:6]) + 1
                need_to_create = True
        if training:
            penname = f"{cellprefix}{pennum:03d}T"
        else:
            penname = f"{cellprefix}{pennum:03d}"

        if need_to_create and create_if_missing:
            self.create_penetration(animal=animal, training=training, penname=penname)

        return penname

    def get_current_site(self, siteid=None, animal=None, training=None, penname=None, create_if_missing=True, force_next=False):
        if animal is None:
            animal = self.animal
        if training is None:
            training = self.training

        if siteid is not None:
            sql = f"SELECT * FROM gCellMaster WHERE siteid='{siteid}' ORDER BY id"
            sitedata = self.pd_query(sql)
            if (len(sitedata) == 0):
                penname = self.get_current_penname(animal, training)
                sitedata = self.pd_query(sql)
                if (len(sitedata) == 0):
                    self.create_site(siteid=siteid)
                    sitedata = self.pd_query(sql)

        else:
            if (penname is None):
                penname = self.get_current_penname(animal, training, create_if_missing=create_if_missing)
            sql = f"SELECT * FROM gCellMaster WHERE penname='{penname}' ORDER BY id"

            sitedata = self.pd_query(sql)
        if (len(sitedata) == 0) & (~create_if_missing):
            sitedata=pd.DataFrame({'penname':penname, 'siteid':penname+'a', 'cellid':penname+'a'},index=[0])
            return sitedata.iloc[-1]

        elif (len(sitedata)==0):
            print(f"No current site for penetration {penname}?")
            return None
        else:
            return sitedata.iloc[-1]

    def create_penetration(self, user=None, animal=None, training=None,
                           penname=None, well=1,
                           NumberOfElectrodes=0, HWSetup=0):
        if user is None:
            user = self.user
        if animal is None:
            animal = self.animal
        if training is None:
            training = self.training
        if penname is None:
            penname = self.get_current_penname(animal, training=training, force_next=True)
        today = datetime.date.today().strftime("%Y-%m-%d")

        racknotes=''
        speakernotes='Free-field Manger'
        probenotes=''
        electrodenotes=''
        ear=''
        print('Creating new penetration')
        d=pd.DataFrame({
            'penname': penname,
            'animal': animal,
            'well': well,
            'pendate': today,
            'who': user,
            'fixtime': datetime.datetime.now().strftime("%H:%M"),
            'ear': ear,
            'rackid': HWSetup,
            'racknotes': racknotes,
            'speakernotes': speakernotes,
            'probenotes': probenotes,
            'electrodenotes': electrodenotes,
            'training': training,
            'numchans': NumberOfElectrodes,
            'addedby': user,
            'info': 'psilbhb.celldb'
        }, index=[0])

        print('Calling sql insert')
        self.sqlinsert('gPenetration', d)
        print(f"Created new penetration {penname}")
        self.create_site(penname=penname, animal=animal,
                         training=training, user=user)
        return penname

    def create_site(self, penname=None, siteid=None, animal=None,
                    training=None, user=None,
                    area="", comments=""):
        if animal is None:
            animal = self.animal
        if training is None:
            training = self.training

        if siteid is None:

            pendata = self.get_penetration(penname).loc[0]
            if user is None:
                user = pendata.addedby
            sitedata = self.get_current_site(penname=penname, force_next=True)
            if sitedata is not None:
                siteletter = sitedata.siteid[-1]
                newsiteletter = chr(ord(siteletter) + 1)
                siteid = sitedata.siteid[:-1] + newsiteletter
            else:
                siteid = penname + 'a'
        else:
            penname = siteid[:-1]
            pendata = self.get_penetration(penname).loc[0]

        d = pd.DataFrame({'siteid': siteid,
                          'cellid': siteid,
                          'penid': pendata.id,
                          'penname': penname,
                          'animal': pendata.animal,
                          'well': pendata.well,
                          'training': pendata.training,
                          'findtime': datetime.datetime.now().strftime("%H:%M"),
                          'area': area,
                          'comments': comments,
                          'addedby': user,
                          'info': 'psilbhb.celldb'}, index=[0])
        masterid = self.sqlinsert('gCellMaster', d)
        print(f'Added gCellMaster entry {siteid}')

        cellid = siteid + "-001-1"
        # sitedata = self.get_current_site(self, penname=penname)
        # masterid = sitedata['id']

        d=pd.DataFrame({'siteid': siteid,
            'cellid': cellid,
            'masterid': masterid,
            'penid': pendata.id,
            'channel': '001-',
            'unit': 1,
            'channum': 1,
            'addedby': user,
            'info': 'psilbhb.celldb'}, index=[0])
        self.sqlinsert('gSingleCell', d)
        print(f'Added gSingleCell entry {cellid}')

        return siteid

    def create_rawfile(self, siteid=None,
                       runclass=None, filenum=0, behavior="passive", timejuice=0,
                       pupil=False, psi_format=True,
                       dataroot='/auto/data/daq/',
                       rawroot=None):
        if siteid is None:
            sitedata = self.get_current_site()
            siteid = sitedata.siteid
        else:
            sitedata = self.get_current_site(siteid=siteid)

        penname = sitedata['penname']
        if runclass is None:
            raise ValueError('Three-letter runclass must be specified for new_raw_file')
        if behavior is None:
            behavior = "passive"

        masterid = sitedata.id
        sql = f"SELECT * FROM gRunClass where name='{runclass}'"
        runclassdata = self.pd_query(sql)
        if len(runclassdata) == 1:
            runclassid = runclassdata.loc[0, 'id']
            stimclass = runclassdata.loc[0, 'stimclass']
            task = runclassdata.loc[0, 'task']
        else:
            raise ValueError(f"runclass {runclass} not found in gRunClass")
        year = datetime.datetime.now().strftime("%Y")
        month = datetime.datetime.now().strftime("%m")
        day = datetime.datetime.now().strftime("%d")
        dataroot = str(dataroot).replace('\\', '/')

        # determine path + parmfilename
        if sitedata.training:
            resppath = f'{dataroot}/{sitedata.animal}/training{year}/'
        else:
            resppath = f'{dataroot}/{sitedata.animal}/{penname}/'
        rawdata = self.get_rawdata(siteid)
        if filenum == 0:
            if len(rawdata) > 0:
                prevparmfile = rawdata.iloc[-1]['parmfile']
                if (filenum==0) and sitedata.training:
                    filenum = int(prevparmfile.split(".")[0].split("_")[-1]) + 1
                elif filenum==0:
                    filenum = int(prevparmfile[len(siteid):(len(siteid) + 2)]) + 1
            else:
                filenum = 1

        if behavior == 'passive':
            bstr = 'p'
        else:
            bstr = 'a'
        if sitedata.training:
            filestr = f'{filenum:d}'
            parmbase = f"{sitedata.animal}_{year}_{month}_{day}_{runclass}_{filestr}"
        else:
            filestr = f'{filenum:02d}'
            parmbase = f"{siteid}{filestr}_{bstr}_{runclass}"
        if parmbase not in list(rawdata['parmfile']):
            pupilfile = ""
            if psi_format:
                parmfile = parmbase
                evpfilename = f"{resppath}{parmbase}/reward_contact_analog.zarr"
                rawpath = f'{resppath}{parmbase}/raw/'
                if pupil:
                    pupilpath = resppath.replace(str(get_config('DATA_ROOT')),
                                                 str(get_config('VIDEO_ROOT')))
                    pupilfile = f"{pupilpath}{parmbase}/recording.avi"

            else:
                parmfile = parmbase + ".m"
                if pupil:
                    pupilfile = f"{resppath}{parmbase}.avi"

                evpfilename = f"{resppath}{parmbase}.evp"
                rawpath = f'{resppath}raw/{siteid}{filestr}/'

            if rawroot is not None:
                resppath = resppath.replace(str(dataroot),str(rawroot).replace('\\','/'))

            d = pd.DataFrame({
                'cellid': siteid,
                'masterid': masterid,
                'runclass': runclass,
                'runclassid': runclassid,
                'training': sitedata.training,
                'resppath': resppath,
                'parmfile': parmfile,
                'eyewin': pupil,
                'eyecalfile': pupilfile,
                'respfileevp': evpfilename,
                'respfile': rawpath,
                'task': task,
                'stimclass': stimclass,
                'behavior': behavior,
                'timejuice': timejuice,
                'fixtime': datetime.datetime.now().strftime("%H:%M"),
                'time': datetime.datetime.now().strftime("%H:%M"),
                'addedby': sitedata.addedby,
                'info': 'psilbhb.celldb'}, index=[0])
            rawid = self.sqlinsert('gDataRaw', d)
            print(f'Added gDataRaw entry {parmbase}')
            rawdata = {'rawid': rawid, 'resppath': resppath, 'parmbase': parmbase,
                       'pupil_file': pupilfile, 'respfile': rawpath,
                       'runclass': runclass, 'evpfilename':evpfilename}
        else:
            rawdata = rawdata.loc[rawdata.parmfile==parmbase].to_dict()

        return rawdata

    def save_data(self, rawid, datadict, parmtype=0, keep_existing=False):

        if len(datadict) == 0:
            log.info('dict={} not saving anything')
            return

        if type(datadict) is not dict:
            raise ValueError(f"Parmeter datadict must be a dict")

        fn = datadict.keys()

        if keep_existing:
            namestr = "'" + "','".join(fn) + "'"

            sql=f'DELETE FROM gData WHERE rawid={rawid}' +\
                f" AND name in ({namestr}) AND parmtype={parmtype}"
        else:
            sql=f'DELETE FROM gData WHERE rawid={rawid} AND parmtype={parmtype}'
        self.sqlexec(sql)

        rawdata = self.get_rawdata(rawid=rawid)
        if len(rawdata) == 0:
            raise ValueError(f"rawid={rawid} not in gDataRaw")

        masterid = rawdata.loc[0,'masterid']
        siteid = rawdata.loc[0,'cellid']
        penid = rawdata.loc[0,'penid']
        d = pd.DataFrame()
        for i,(k,v) in enumerate(datadict.items()):
            jv = json.dumps(v, cls=PSIJsonEncoder)
            d.loc[i,'name']=k
            d.loc[i,'svalue']=jv
        d['siteid']=siteid
        d['penid']=masterid
        d['masterid']=masterid
        d['rawid']=rawid
        d['datatype']=1
        d['parmtype']=parmtype
        d['addedby']=self.user
        d['info']='psilbhb.celldb'

        self.sqlinsert('gData', d)

        log.info(f'Saved {len(d)} data items for rawid {rawid}')

    def read_data(self, rawid):
        sql = f"SELECT * FROM gData WHERE rawid={rawid} ORDER BY parmtype,id"

        d = self.pd_query(sql)
        sidx = d['datatype'] == 1
        d['value'] = d['value'].astype(object)
        #d.loc[sidx, 'value'] = d.loc[sidx, 'svalue']

        try:
            d.loc[sidx,'value'] = d.loc[sidx,'svalue'].apply(json.loads)
        except:
            pass
        try:
            sidx = np.isnan(d['value'].astype(float))
            d.loc[sidx, 'value'] = d.loc[sidx, 'svalue']
        except:
            pass
        return d

    def refresh_rawdata(self, rawid):
        if self.user is None:
            self.user='lbhb'
        rawdata = self.get_rawdata(rawid=rawid)
        rawdata['parmbase']=rawdata['parmfile']
        rawdata['rawid']=rawdata['id']
        rawdata = rawdata.loc[0]
        filename = os.path.join(rawdata['resppath'],rawdata['parmbase'])
        if not os.path.exists(filename + '/trial_log.csv'):
            filename = filename.replace('d:','e:')
            filename = filename.replace('/auto/data/', 'h:/')
        d, dataparm, dataperf = readpsievents(filename, rawdata['runclass'])

        self.sqlupdate('gDataRaw', rawdata['rawid'], d=d, idfield='id')
        self.save_data(rawdata['rawid'], dataparm, parmtype=0, keep_existing=False)
        self.save_data(rawdata['rawid'], dataperf, parmtype=1, keep_existing=False)

    def fix_rawdata(self, sitemask="", user='david'):
        if self.user is None:
            self.user=user
        sql = f"SELECT * FROM gDataRaw WHERE not(bad) AND isnull(trials) AND cellid like '{sitemask}%'"
        rawdata = self.pd_query(sql)
        rawids=[]
        for i,r in rawdata.iterrows():
            print(i,r['id'],r['resppath'])
            self.refresh_rawdata(r['id'])
            rawids.append(r['id'])
        return rawids

def __main__():

    c = celldb()
    c.animal = "Test"
    c.training = True
    c.user = "david"
    penname = c.get_current_penname()
    sitedata = c.get_current_site()

    #resppath, parmbase, rawpath, rawid = c.create_rawfile(runclass="TSP", behavior="active", pupil=True)
    #resppath, parmbase, rawpath, rawid = c.create_rawfile(runclass="NON", behavior="passive", pupil=False)

    #c.sqlupdate('gDataRaw', rawid, {'trials': 60, 'corrtrials': 50})

    rawid = 146632
    datadict={'Trial_Paradigm': 'GoNogo',
              'Trial_CatchFrac': 0.5,
              'Trial_Durations': [1.0, 2.0, 3.0],
              'ReferenceClass': 'NaturalSounds',
              'TargetClass': 'Tone',
              'Tar_Frequency': 1000}
    parmtype=0
    #c.save_data(rawid, datadict, parmtype=parmtype)

    rawid = 146342
    rawid = 146632

    d=c.read_data(rawid)
    print(d[['name','value']])


def flush_training(prefix="LMD", local_folder="e:/data", dest_root='/auto/data/daq',
                   dest_root_win='h:/daq', c=None):
    if c is None:
        c = celldb()
    sql = f"SELECT * FROM gDataRaw WHERE not(bad) AND cellid like '{prefix}%' AND respfile LIKE '{local_folder}%'"
    print(sql)
    df_to_move = c.pd_query(sql)
    print(f"Found {len(df_to_move)} files to flush")
    for i, r in df_to_move.iterrows():
        dataroot, f = os.path.split(r['respfile'])
        dataroot, f = os.path.split(dataroot)
        destpath = dataroot.replace(local_folder, dest_root)
        destpath_win = dataroot.replace(local_folder, dest_root_win)
        print(f"Copying files {dataroot} --> {destpath_win}")
        shutil.copytree(dataroot, destpath_win, dirs_exist_ok=True)

        print(f"Updating paths in celldb")
        sql = f"UPDATE gDataRaw SET" +\
              f" respfileevp=replace(respfileevp, '{local_folder}', '{dest_root}')," + \
              f" respfile=replace(respfile, '{local_folder}', '{dest_root}')," +              \
              f" eyecalfile=replace(eyecalfile, '{local_folder}', '{dest_root}')," +              \
              f" resppath=replace(resppath, 'd:/Data', '{dest_root}')" +\
              f" WHERE id={r['id']}"
        c.sqlexec(sql)

    return df_to_move





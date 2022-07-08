import datetime

from sqlalchemy import create_engine, desc, exc
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.automap import automap_base

import pandas as pd
import pandas.io.sql as psql

MYSQL_USER='XXX'
MYSQL_PASS='XXX'
MYSQL_HOST='XX.XX.XX'
MYSQL_PORT='3306'
MYSQL_DB='XXX'

class celldb():
    
    ENGINE = None
    user = None
    animal = None
    
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
                MYSQL_USER, MYSQL_PASS, MYSQL_HOST,
                MYSQL_PORT, MYSQL_DB
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
            d = pd.read_sql_query(sql=sql, con=engine, params=params)
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
        conn.execute(sql)

        return True

    def sqlinsert(self, table, d):
        """
        TODO: what if d is too long?
        """
        engine = self.Engine()
        conn = engine.connect()

        # creating column list for insertion
        cols = "`,`".join([str(i) for i in d.columns.tolist()])

        # Insert DataFrame in a bulk insert
        sql = f"REPLACE INTO `{table}` (`" +cols + "`) VALUES (" + "%s,"*(len(d.columns.tolist())-1) + "%s)"
        rows=[]
        for i,row in d.iterrows():
            rows.append(tuple(row.values))

        conn.execute(sql, rows)

    def get_users(self):
        d = self.pd_query("SELECT id,userid,email FROM gUserPrefs WHERE active")
        return d

    def get_animals(self, name=None, lab="lbhb", species="ferret"):
        if name is None:
            d = self.pd_query("SELECT * FROM gAnimal WHERE onschedule<2 AND lab=%s AND species=%s", (lab,species))
        else:
            d = self.pd_query("SELECT * FROM gAnimal WHERE animal=%s", (name,))

        return d

    def get_next_penname(self, animal, training=1):

        d_animal = self.get_animals(name=animal)
        if len(d_animal)>0:
            cellprefix = d_animal.loc[0, 'cellprefix']
        else:
            raise ValueError(f"animal {animal} not found in celldb.gAnimal")

        sql = 'SELECT max(id) as maxid FROM gPenetration WHERE training=%s AND animal=%s'
        lastpendata = self.pd_query(sql, (training, animal))

        if len(lastpendata) == 0:
            print('No penetrations exist for this animal. Guessing info from scratch.')
            pennum=1
        else:
            sql = f"SELECT * FROM gPenetration WHERE id={lastpendata.loc[0,'maxid']}"
            lastpendata = self.pd_query(sql)
            penname = lastpendata.loc[0,'penname']
            print(f"Inferring info from pen {penname}")
            pennum = int(penname[3:6])+1

        if training:
            penname = f"{cellprefix}{pennum:03d}T"
        else:
            penname = f"{cellprefix}{pennum:03d}"

        return penname

    def create_penetration(self, user=None, animal=None, penname=None, well=1,
                           NumberOfElectrodes=0, training=1, HWSetup=0):
        if user is None:
            user = self.user
        if animal is None:
            animal = self.animal
        if penname is None:
            penname = self.get_next_penname(animal,training=training)

        today = datetime.date.today().strftime("%Y-%m-%d")

        racknotes='TEST'
        speakernotes='Free-field Manger'
        probenotes='TEST'
        electrodenotes="TEST"
        ear='TEST'
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
            'addedby': user,
            'info': 'psilbhb.celldb'
        }, index=[0])
        return d

    def get_current_penetration(self, animal=None):
        if animal is None:
            animal = self.animal
            
        return penname

    def get_current_site(self, animal, penname=None):

        return siteid

    def new_raw_file(self, siteid, runclass, behavior):

        return 1


"""
% function penid=dbcreatepen(globalparams)                                                                                                                                          
%                                                                                                                                                                           
% created SVD 2005-11-21                                                                                                                                                    
%                                                                                                                                                                           
function penid=dbcreatepen(globalparams)                                                                                                                                    
                                                                                                                                                                            
dbopen;                                                                                                                                                                     
                                                                                                                                                                            
sql=sprintf('SELECT * FROM gAnimal WHERE animal like "%s"',...                                                                                                              
            globalparams.Ferret);                                                                                                                                
adata=mysql(sql);                                                                                                                                                
if length(adata)==0,                                                                                                                                             
   error('Ferret not found in cellDB!');                                                                                                                         
end                                                                                                                                                              
globalparams.animal=adata(1).animal;                                                                                                                             
                                                                                                                                                                 
sql=sprintf('SELECT * FROM gUserPrefs WHERE realname like "%%%s%%"',...                                                                                          
            globalparams.Tester);                                                                                                                                
udata=mysql(sql);                                                                                                                                                
if length(udata)==0,                                                                                                                                             
    sql=sprintf('SELECT * FROM gUserPrefs WHERE userid like "%%%s%%"',...                                                                                        
            globalparams.Tester);                                                                                                                                
    udata=mysql(sql);                                                                                                                                            
end                                                                                                                                                              
if length(udata)==0,                                                                                                                                             
   error('Tester not found in cellDB!');                                                                                                                         
end
globalparams.who=udata(1).userid;

sql=['SELECT max(id) as maxid FROM gPenetration' ...
     ' WHERE training in (0,1) AND animal="',globalparams.animal,'"'];
lastpendata=mysql(sql);

if isempty(lastpendata.maxid),
   
   warning('No penetrations exist for this animal. Guessing info from scratch.');
   
   if ~isfield(globalparams,'well'),
      globalparams.well=1;
   end
   globalparams.cellprefix=adata(1).cellprefix;
   globalparams.eye='';
   globalparams.mondist=0;
   globalparams.etudeg=0;
else
   
   sql=['SELECT gPenetration.*, gAnimal.cellprefix FROM gPenetration'...
        ' INNER JOIN gAnimal ON gAnimal.animal=gPenetration.animal'...
        ' WHERE gPenetration.id=',num2str(lastpendata.maxid)];
   lastpendata=mysql(sql);
   
   fprintf('Guessing info from pen %s\n', lastpendata.penname);

   if ~isfield(globalparams,'well'),
      globalparams.well=lastpendata.well;
   end
   globalparams.cellprefix=lastpendata.cellprefix;
   globalparams.eye=lastpendata.eye;
   globalparams.mondist=lastpendata.mondist * 1.0;
   globalparams.etudeg=lastpendata.etudeg * 1.0;
end

globalparams.numchans=globalparams.NumberOfElectrodes;
if strcmp(globalparams.Physiology,'No'),
    globalparams.training=1;
else
    globalparams.training=0;
end
globalparams.probenotes='';
globalparams.electrodenotes='';

% Log hardware setup-specific information
try
    HWSetupSpecs=BaphyMainGuiItems('HWSetupSpecs',globalparams);
    ff=fields(HWSetupSpecs);
    for ii=1:length(ff),
        globalparams.(ff{ii})=HWSetupSpecs.(ff{ii});
    end
catch
    % this information should be moved to <config>\<lab>\BaphyMainGuiItems
    % case 'HWSetupSpecs'
    switch globalparams.HWSetup,
        case 0,
            globalparams.racknotes='TEST MODE';
            globalparams.speakernotes='';
            globalparams.probenotes='';
            globalparams.ear='';
        case 1,
            globalparams.racknotes=sprintf('Soundproof room 1, pump cal: %.2f ml/sec',globalparams.PumpMlPerSec.Pump);
            globalparams.speakernotes='Etymotic earphone. Calibrated: Krohn-Hite filter, Rane equalizer, HP attenuator, Rane amplifier. (2007-09-24)';
            if ~globalparams.training,
                globalparams.probenotes=sprintf('%d-channel. Well position: XXX',globalparams.numchans);
                globalparams.electrodenotes='FHC: size, impendence not specified';
            end
            globalparams.ear='R';
        case 2,
            globalparams.racknotes=sprintf('Training rig 1, pump cal: %.2f ml/sec',globalparams.PumpMlPerSec.Pump);
            globalparams.speakernotes='Free field, front facing.  Crown amplifier. (2006-04-28)';
            globalparams.ear='B';
        case 3,
            globalparams.racknotes=sprintf('Soundproof room 2, pump cal: %.2f ml/sec',globalparams.PumpMlPerSec.Pump);
            globalparams.speakernotes='Etymotic earphone.  Calibrated: Krohn-Hite filter, Rane equalizer, HP attenuator, Radio Shack amplifier. (2006-04-28)';
            if ~globalparams.training,
                globalparams.probenotes=sprintf('%d-channel. Well position: XXX',globalparams.numchans);
                globalparams.electrodenotes='FHC: size, impendence not specified';
            end
            globalparams.ear='L';
        case 4,
            globalparams.racknotes=sprintf('Training rig 2, pump cal: %.2f ml/sec',globalparams.PumpMlPerSec.Pump);
            globalparams.speakernotes='Free field, front facing. Onkyo amplifier. (2006-04-28)';
            globalparams.ear='B';
        case 5,
            globalparams.racknotes=sprintf('Holder training booth 1, pump cal: %.2f ml/sec',globalparams.PumpMlPerSec.Pump);
            globalparams.speakernotes='Etymotic earphone.  Calibrated: Krohn-Hite filter, Rane equalizer, HP attenuator, Rane amplifier. (2007-02-10)';
            if ~globalparams.training,
                globalparams.probenotes=sprintf('%d-channel. Well position: XXX',globalparams.numchans);
                globalparams.electrodenotes='FHC: size, impendence not specified';
            end
            globalparams.ear='L';
        otherwise,
            globalparams.racknotes='UNKNOWN';
            globalparams.speakernotes='UNKNOWN';
            globalparams.ear='';
    end
end

sql=['SELECT * FROM gPenetration'...
     ' WHERE animal="',globalparams.animal,'"',...
     ' AND pendate="',globalparams.date,'" AND training=2'];
wdata=mysql(sql);

if length(wdata)>0,
   penid=wdata(1).id;
   sql=['UPDATE gPenetration SET',...
        ' penname="',globalparams.penname,'",',...
        'animal="',globalparams.animal,'",',...
        'well=',num2str(globalparams.well),',',...
        'pendate="',globalparams.date,'",',...
        'who="',globalparams.who,'",',...
        'fixtime="',datestr(now,'HH:MM'),'",',...
        'ear="',globalparams.ear,'",',...
        'numchans=',num2str(globalparams.numchans),',',...
        'rackid=',num2str(globalparams.HWSetup),',',...
        'racknotes="',globalparams.racknotes,'",',...
        'speakernotes="',globalparams.speakernotes,'",',...
        'probenotes="',globalparams.probenotes,'",',...
        'electrodenotes="',globalparams.electrodenotes,'",',...
        'training=',num2str(globalparams.training),',',...
        'addedby="',globalparams.who,'",',...
        'info="dbcreatepen.m"',...
        ' WHERE id=',num2str(penid)];
   mysql(sql);
   fprintf('updated gPenetration entry %d\n',penid);
else
   [aff,penid]=sqlinsert('gPenetration',...
                         'penname',globalparams.penname,...
                         'animal',globalparams.animal,...
                         'well',globalparams.well,...
                         'pendate',globalparams.date,...
                         'who',globalparams.who,...
                         'fixtime',datestr(now,'HH:MM'),...
                         'ear',globalparams.ear,...
                         'numchans',globalparams.numchans,...
                         'rackid',globalparams.HWSetup,...
                         'racknotes',char(globalparams.racknotes),...
                         'speakernotes',char(globalparams.speakernotes),...
                         'probenotes',char(globalparams.probenotes),...
                         'electrodenotes',char(globalparams.electrodenotes),...
                         'training',globalparams.training,...
                         'addedby',globalparams.who,...
                         'info','dbcreatepen.m');
   fprintf('added gPenetration entry %d\n',penid);
end

% function rawid=dbcreateraw(globalparams,runclass,mfilename,evpfilename)
%
% created SVD 2005-11-21
%
function rawid=dbcreateraw(globalparams,runclass,mfilename,evpfilename)

dbopen;

%check to see if run class is valid
underscores=strfind(runclass,'_');
if(isempty(underscores))
sql=sprintf('SELECT * FROM gRunClass WHERE name="%s"',runclass);
rdata=mysql(sql);
if length(rdata)==0;
   error('runclass not found in cellDB!');
end
else
    sts=[1 underscores+1];
    underscores(end+1)=length(runclass)+1;
    runclass2=[runclass,','];
    for i=1:length(underscores)
        sql=sprintf('SELECT * FROM gRunClass WHERE name="%s"',runclass2(sts(i):underscores(i)-1));
        rdata=mysql(sql);
        if length(rdata)==0;
            error('runclass not found in cellDB!');
        end
    end
end

sql=sprintf('SELECT * FROM gCellMaster WHERE id=%d',globalparams.masterid);
sdata=mysql(sql);
if length(sdata)==0,
   error('Site not found in cellDB!');
end
sql=sprintf('SELECT * FROM gSingleCell WHERE masterid=%d',globalparams.masterid);
celldata=mysql(sql);
if length(celldata)==0,
   error('Cell not found in cellDB!');
end

[respfile,resppath]=basename(mfilename);

% avoid losing backslashes in SQL
% don't need to do this since it's taken care of in mysql.m
%resppath=strrep(resppath,'\','\\');
%evpfilename=strrep(evpfilename,'\','\\');

if strcmp(globalparams.Physiology,'No'),
    behavior='active';
elseif strcmp(globalparams.Physiology,'Yes -- Passive'),
    behavior='passive';
elseif strcmp(globalparams.Physiology,'Yes -- Behavior'),
    behavior='active';
end

[aff,rawid]=sqlinsert('gDataRaw',...
                      'cellid',globalparams.SiteID,...
                      'masterid',globalparams.masterid,...
                      'runclass',runclass,...
                      'runclassid',rdata.id,...
                      'task',rdata.task,...
                      'training',sdata.training,...
                      'respfileevp',evpfilename,...
                      'respfile','*SAVE MAP FILE NAME HERE*',...
                      'parmfile',respfile,...
                      'resppath',resppath,...
                      'fixtime',datestr(now,'HH:MM'),...
                      'behavior',behavior,...
                      'stimclass',rdata.stimclass,...
                      'time',datestr(now,'HH:MM'),...
                      'timejuice',globalparams.PumpMlPerSec.Pump,...
                      'addedby',sdata.addedby,...
                      'info','dbcreatepen.m');
fprintf('added gDataRaw entry %d\n',rawid);

[aff,singlerawid]=sqlinsert('gSingleRaw',...
                         'cellid',celldata(1).cellid,...
                         'masterid',globalparams.masterid,...
                         'singleid',celldata(1).id,...
                         'penid',globalparams.penid,...
                         'rawid',rawid,...
                         'channel','a',...
                         'unit',1,...
                         'channum',1,...
                         'addedby',sdata.addedby,...
                         'info','dbcreatepen.m');
fprintf('added gSingleRaw entry %d\n',singlerawid);

% function masterid=dbcreatesite(params)
%
% created SVD 2005-11-21
%
function masterid=dbcreatesite(params)

dbopen;

sql=sprintf('SELECT * FROM gPenetration WHERE id=%d',params.penid);
pdata=mysql(sql);
if length(pdata)==0,
   error('Penetration not found in cellDB!');
end

if pdata.numchans > 99,
   cellid=[params.SiteID,'-001-1'];
elseif pdata.numchans > 8,
   cellid=[params.SiteID,'-01-1'];
else
   cellid=[params.SiteID,'-a1'];
end

[aff,masterid]=sqlinsert('gCellMaster',...
                         'siteid',params.SiteID,...
                         'cellid',params.SiteID,...
                         'penid',params.penid,...
                         'penname',pdata.penname,...
                         'animal',pdata.animal,...
                         'well',pdata.well,...
                         'training',pdata.training,...
                         'findtime',datestr(now,'HH:MM'),...
                         'addedby',pdata.who,...
                         'info','dbcreatepen.m');
fprintf('added gCellMaster entry %d\n',masterid);

[aff,singleid]=sqlinsert('gSingleCell',...
                         'siteid',params.SiteID,...
                         'cellid',cellid,...
                         'masterid',masterid,...
                         'penid',params.penid,...
                         'channel','a',...
                         'unit',1,...
                         'channum',1,...
                         'addedby',pdata.who,...
                         'info','dbcreatepen.m');
fprintf('added gSingleCell entry %d\n',singleid);

% function siteid=dbgetlastsite(Ferret, doingphysiology);
function siteid=dbgetlastsite(Ferret, doingphysiology);

if ~dbopen,
   siteid='';
   return;
end
sql=sprintf('SELECT * FROM gAnimal WHERE animal like "%s"',Ferret);
adata=mysql(sql);
if length(adata)==0,
    warning('%s not in celldb\n',Ferret);
    siteid='';
    return;
end
animal=adata(1).animal;

% find most recent penetration and time that last site was updated in DB
sql=['SELECT gPenetration.id as maxid,',...
     ' time_to_sec(timediff(now(),gCellMaster.lastmod))/86400 as daylag',...
     ' FROM gPenetration,gCellMaster',...
     ' WHERE gPenetration.id=gCellMaster.penid',...
     ' AND gPenetration.training=',num2str(1-doingphysiology),...
     ' AND gPenetration.animal="',animal,'"'...
     ' ORDER BY gPenetration.id DESC,gCellMaster.id DESC LIMIT 1'];
lastpendata=mysql(sql);
if length(lastpendata)>0,
    daylag=lastpendata.daylag;
else
    daylag=1; % force new site
end

if length(lastpendata)==0 || isempty(lastpendata.maxid) | strcmp(lastpendata.maxid,'NULL'),

    warning(['No penetrations exist for this animal. Guessing SiteID' ...
        ' from scratch.']);
    if doingphysiology,
        siteid=[adata(1).cellprefix,'001a'];
    else
        siteid=[adata(1).cellprefix,'001Ta'];
    end
    return
else
    sql=['SELECT penid,max(siteid) as siteid FROM gCellMaster'...
        ' WHERE penid=',num2str(lastpendata.maxid),' GROUP BY penid'];
    tpendata=mysql(sql);

    if length(tpendata)==0,
        siteid='';
    else
        siteid=tpendata.siteid;
    end
    if isempty(siteid),
        % no sites for this penetration yet
        sql=['SELECT * FROM gPenetration'...
            ' WHERE id=',num2str(lastpendata.maxid)];
        lastpendata=mysql(sql);
        if doingphysiology,
            siteid=[lastpendata.penname,'a'];
        else
            siteid=[lastpendata.penname,'Ta'];
        end
    end

    % if more than 12 hours since last change to gCellMaster, assume that
    % this is a new site.
    if daylag>=0.5,
        numidx=find(siteid>='0' & siteid<='9');
        pennum=str2num(siteid(numidx));

        if doingphysiology,
            siteid=[siteid(1:numidx(1)-1) sprintf('%03d',pennum+1) 'a'];
        else
            siteid=[siteid(1:numidx(1)-1) sprintf('%03d',pennum+1) 'Ta'];
        end
        %fprintf('guessing new penetration/site: %s\n',siteid);
    end
end
% function [parm,perf]=dbReadData(rawid);
%
% created SVD 2006-02-23
%
function [parm,perf]=dbReadData(rawid);

if ~exist('rawid','var'),
   error('parameter rawid required');
end

dbopen;
global DB_USER

sql=['SELECT * FROM gData WHERE rawid=',num2str(rawid),...
     ' ORDER BY id'];
data=mysql(sql);

parm=[];
perf=[];

for ii=1:length(data),
   switch data(ii).datatype,
    case 0,
     val=data(ii).value;
     case 1,
       try,
           val=eval(data(ii).svalue);
       catch
           val=eval([data(ii).svalue ' ]']);
       end
    case 2,
     val=data(ii).svalue;
   end
   
   if data(ii).parmtype==0,
      parm=setfield(parm,data(ii).name,val);
   else
      perf=setfield(perf,data(ii).name,val);
   end
end

% function dbWriteData(rawid,data,parmtype,keep_existing);
%
% rawid - index into gDataRaw table (stored in globalparams.rawid)
% data - structure with each field either a scalar, matrix or string
% parmtype - 'parm' (or 0) - parameters
%            'perf' (of 1) - performance data
% keep_existing - if 1, keep existing data for this rawid (default 0)
%
% created SVD 2006-02-23
%
function dbWriteData(rawid,data,parmtype,keep_existing);

dbopen;
global DB_USER

if ~exist('parmtype','var'),
   parmtype=0;
end
if ~exist('keep_existing','var'),
   keep_existing=0;
end
if ~isnumeric(parmtype) & strcmp(lower(parmtype),'parm'),
   parmtype=0;
elseif ~isnumeric(parmtype) & strcmp(lower(parmtype),'perf'),
   parmtype=1;
elseif ~isnumeric(parmtype),
   parmtype=0;
end

if ~isstruct(data),
   ff=inputname(2);
   if isempty(ff),
      ff='data';
   end
   td=struct(ff,data);
   data=td;
end

fn=fieldnames(data);

if keep_existing,
    % only delete entries that are getting replaced
    namestr='(';
    for ii=1:length(fn),
        namestr=[namestr,'"',fn{ii},'",'];
    end
    namestr(end)=')';
    
    sql=['DELETE FROM gData WHERE rawid=',num2str(rawid),...
        ' AND name in ',namestr,...
        ' AND parmtype=',num2str(parmtype)];
    mysql(sql);
else
    sql=['DELETE FROM gData WHERE rawid=',num2str(rawid),...
        ' AND parmtype=',num2str(parmtype)];
    mysql(sql);
end

rawdata=mysql(['select * FROM gDataRaw WHERE id=', ...
               num2str(rawid)]);

if length(rawdata)==0,
   error(['gRawData.id=',num2str(rawid),' does not exist.']);
end


sql_st=['INSERT INTO gData (masterid,rawid,name,value,svalue,' ...
     'datatype,parmtype,addedby,info) VALUES '];
for ii=1:length(fn),
   val=getfield(data,fn{ii});

   if isnumeric(val) && length(val)==1 && ~isnan(val) && ~isinf(val),
      sql2{ii}=sprintf('(%d,%d,"%s",%f,NULL,0,%d,"%s","dbWriteData.m")',...
                   rawdata.masterid,rawid,fn{ii},val,parmtype,...
                   DB_USER);
      
   elseif isnumeric(val),
      ss=mat2str(val);
      sql2{ii}=sprintf('(%d,%d,"%s",NULL,"%s",1,%d,"%s","dbWriteData.m")',...
                   rawdata.masterid,rawid,fn{ii},ss,parmtype,...
                   DB_USER);
   elseif isstruct(val),
       if ~strcmp(fn{ii},'Behave_DisplayParams')
           error('Count not save %s, fix!',fn{ii}')
       end
   elseif ~iscell(val)
      sql2{ii}=sprintf('(%d,%d,"%s",NULL,"%s",2,%d,"%s","dbWriteData.m")',...
                   rawdata.masterid,rawid,fn{ii},val,parmtype,...
                   DB_USER);
   elseif iscell(val)
       val=join(val);
       val=val{1};
       sql2{ii}=sprintf('(%d,%d,"%s",NULL,"%s",2,%d,"%s","dbWriteData.m")',...
                   rawdata.masterid,rawid,fn{ii},val,parmtype,...
                   DB_USER);
   end
end
sql=sql_st;
for ii=1:length(fn),
    if ~isempty(sql2{ii})
        sql=[sql sql2{ii} ','];
    end
end
% remove last comma
sql=sql(1:end-1);
try
    mysql(sql);
catch err
    if length(err.message)>59 && strcmp(err.message(1:59),'Error running external mysql: The command line is too long.')
        inds=floor(linspace(1,length(fn),3)); inds(end)=inds(end)+1;
        for j=1:length(inds)-1
            sql=sql_st;
            for ii=inds(j):(inds(j+1)-1)
                if ~isempty(sql2{ii})
                    sql=[sql sql2{ii} ','];
                end
            end
            sql=sql(1:end-1);% remove last comma
            mysql(sql);
        end
    else
        rethrow(err)
    end
end
fprintf('Saved %d data items for rawid %d\n',length(fn),rawid);

"""

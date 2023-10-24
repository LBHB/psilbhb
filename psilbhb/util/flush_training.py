
import os
import shutil

from psilbhb.util.celldb import celldb, flush_training

def flush_training_old(prefix="LMD", local_folder="e:/data", dest_root='/auto/data/daq',
                   dest_root_win='h:/daq'):
    """
    :param prefix:  default is 'LMD'
    :param local_folder:  (Windows -- 'e:/data' for badger psilbhb)
    :param dest_root:  (Linux file root -- '/auto/data/daq')
    :param dest_root_win:  (Windows root -- 'h:/daq')
    :return: df_to_move: dataframe of flushed files
    """
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


flush_training("LMD")
flush_training("SQD")
flush_training("SDS")
flush_training("SLJ")


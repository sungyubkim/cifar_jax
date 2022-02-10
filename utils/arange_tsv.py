import os
import numpy as np
import pandas as pd
from absl import app, flags
from utils.ckpt import dict2tsv, check_dir

flags.DEFINE_string('dataset', 'cifar10', help='dataset')
FLAGS = flags.FLAGS

def generate_trial_path(train_setting:str, trial:int):
    trial_path = train_setting.split('_')
    trial_path[-1] = str(trial)
    return '_'.join(trial_path)

def search(key):
    res_path = f'./res_cifar/{FLAGS.dataset}'
    result = []
    for paths, dirs, files in os.walk(res_path):
        for f in files:
            if key in f:
                f_path = f'{paths}/{f}'
                result.append(f_path)
    return result

def reduce_random_seed(log_files:list):
    res_path = f'./res_cifar/{FLAGS.dataset}'
    log_path = f'{res_path}/merged_log'
    check_dir(log_path)
    for log_f in log_files:
        train_setting = '/'.join(log_f.split('/')[3:])
        random_seed = train_setting.split('_')[-1][0]
        if int(random_seed) == 0:
            # merge multiple random seed
            df_list = []
            for i in range(4):
                trial_path = generate_trial_path(train_setting, i)
                if os.path.exists(f'{res_path}/{trial_path}/sharpness.tsv'):
                    trial_log = pd.read_csv(f'{res_path}/{trial_path}/sharpness.tsv', sep='\t').to_dict(orient='records')[-1]
                    df_list.append(trial_log)
            df = pd.DataFrame(df_list)
            # average multiple random seed
            new_df = {'train_setting' : f'{train_setting}'}
            for k,v in df.items():
                new_df[f'{k} m'] = f'{np.mean(v):.4f}'
                new_df[f'{k} s'] = f'{np.std(v):.4f}'
            merged_path = train_setting.split('_')
            merged_path.pop(1) # remove random seed number
            merged_path.pop(-2) # remove sigma
            merged_path.pop(-2) # remove lambda
            merged_path = '_'.join(merged_path)
            merged_path = '_'.join(merged_path.split('/'))
            merged_path = f'{log_path}/{merged_path}'
            dict2tsv(new_df, merged_path)
    return None

def main(_):
    log_files = search('sharpness')
    reduce_random_seed(log_files)
    return None

if __name__=='__main__':
    app.run(main)
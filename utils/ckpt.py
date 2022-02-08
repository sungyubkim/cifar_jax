import os
import jax
from flax.training import checkpoints

def check_dir(folder_path):
    # save path
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def dict2tsv(res, file_name):
    for k, v in res.items():
        if type(v) == float:
            res[k] = '{:.4e}'.format(v)
    if not os.path.exists(file_name):
        with open(file_name, 'a') as f:
            f.write('\t'.join(list(res.keys())))
            f.write('\n')

    with open(file_name, 'a') as f:
        f.write('\t'.join([str(r) for r in list(res.values())]))
        f.write('\n')

def save_ckpt(state, path):
    if jax.process_index() == 0:
        state = jax.device_get(jax.tree_map(lambda x: x[0], state))
        checkpoints.save_checkpoint(path, state, state.step, overwrite=True)
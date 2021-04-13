import os
max_seq_length = 128
num_classes = 2
num_train_data = 174449
import tensorflow as tf

# flags = tf.flags
# flags.DEFINE_string("dataset", "nba", "The data config.")

dataset_dir = 'e2e_data'

# load all entities
# if 'nba' in flags.config_data:
#     dataset_dir = 'nba_data'
# elif 'e2e' in flags.config_data:
#     dataset_dir = 'e2e_data'
# else:
#     print('[info] You need to choose one dataset.')

modes = ['train', 'val', 'test']
mode_to_filemode = {
    'train': 'train',
    'val': 'valid',
    'test': 'test',
}
field_to_vocabname = {
    'x_value': 'x_value',
    'x_type': 'x_type',
    'x_associated': 'x_associated',
    'y_aux': 'y',
    'x_ref_value': 'x_value',
    'x_ref_type': 'x_type',
    'x_ref_associated': 'x_associated',
    'y_ref': 'y',
}
fields = list(field_to_vocabname.keys())
train_batch_size = 32
eval_batch_size = 32
batch_sizes = {
    'train': train_batch_size,
    'val': eval_batch_size,
    'test': eval_batch_size,
}

datas = {
    mode: {
        'num_epochs': 1,
        'shuffle': mode == 'train',
        'batch_size': batch_sizes[mode],
        'allow_smaller_final_batch': mode != 'train',
        'datasets': [
            {
                'files': [os.path.join(
                    dataset_dir, mode,
                    '{}.{}.txt'.format(field, mode_to_filemode[mode])
                )],
                'vocab_file': os.path.join(
                    dataset_dir,
                    '{}.vocab.txt'.format(field_to_vocabname[field])),
                'data_name': field,
            }
            for field in fields]
    }
    for mode in modes
}

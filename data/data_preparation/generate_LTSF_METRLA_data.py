import os
import sys
import numpy as np
import pandas as pd


# Hyper-parameters
dataset_name = 'METRLA'
data_file_path = f'../raw_data/{dataset_name}/{dataset_name}.h5'
graph_file_path = f'../raw_data/{dataset_name}/adj_{dataset_name}.pkl'
output_dir = os.path.join('../datasets_zoo/', dataset_name)
target_channel = [0] # Target traffic flow channel
add_time_of_day = True  # Add time of day as a timestamp feature
add_day_of_week = True  # Add day of the week as a timestamp feature
steps_per_day = 288  # Number of time steps per day
days_per_week = 7 # Number of days per week
frequency = 1440 // steps_per_day
domain = 'traffic speed'
feature_description = [domain, 'time of day', 'day of week']
regular_settings = {
    'IN_STEPS': 96,
    'OUT_STEPS': 96,
    'TRAIN_VAL_TEST_RATIO': [0.7, 0.1, 0.2],
    'SPLIT': 'LTSF'
}


def load_and_preprocess_data(save_data:bool):

    save_data = save_data

    split = regular_settings['SPLIT']
    in_steps = regular_settings['IN_STEPS']
    out_steps = regular_settings['OUT_STEPS']
    train_ratio = regular_settings['TRAIN_VAL_TEST_RATIO'][0]
    val_ratio = regular_settings['TRAIN_VAL_TEST_RATIO'][1]
    
    if data_file_path.endswith('h5'):

        file_type = 'hdf'
        df = pd.read_hdf(data_file_path)
        data = np.expand_dims(df.values, axis=-1)

    elif data_file_path.endswith('npz'):

        file_type = 'npz'
        data = np.load(data_file_path)['data']

    elif data_file_path.endswith('csv'):

        file_type = 'csv'
        df = pd.read_csv(data_file_path)
        df_index = pd.to_datetime(df['date'].values, format='%Y-%m-%d %H:%M:%S').to_numpy()
        df = df[df.columns[1:]]
        df.index = df_index
        data = np.expand_dims(df.values, axis=-1)

    else: 
        raise TypeError('Unsupported data file type.')
    
    data = data[..., target_channel] # [all_steps, num_nodes, num_channels]
    print('Raw traffic series shape: {0}'.format(data.shape))

    L, N, F = data.shape
    if split in ['DEFAULT', 'STF']:
        """
        Default setting: first sliding window, then split.
            - This is not strict as it will cross the boundaries of train&val, val&test,
            - Besides, it generates more samples.
        """
        num_samples = L - (in_steps + out_steps) + 1
        train_num_short = round(num_samples * train_ratio)
        val_num_short = round(num_samples * val_ratio)
        test_num_short = num_samples - train_num_short - val_num_short

        index_list = [
            (t - in_steps, t, t + out_steps) for t in range(in_steps, num_samples + in_steps)
            ]
        train_index = index_list[:train_num_short]
        val_index = index_list[train_num_short:train_num_short+val_num_short]
        test_index = index_list[train_num_short +
                                val_num_short : train_num_short + val_num_short + test_num_short]  
        
    elif split == 'STRICT':
        """
        Disable overlapping: first split train/val/test, then perform sliding window individually
        It generate least samples.
        """
        split1 = round(L * train_ratio)
        split2 = round(L * (train_ratio + val_ratio))
        train_index = [
            (t - in_steps, t, t + out_steps) for t in range(in_steps, split1 - out_steps + 1)
            ]
        val_index = [
            (t - in_steps, t, t + out_steps) for t in range(split1 + in_steps, split2 - out_steps + 1)
            ]
        test_index = [
            (t - in_steps, t, t + out_steps) for t in range(split2 + in_steps, L - out_steps + 1)
            ]
        
    elif split == 'LTSF':
        """
        According to https://github.com/cure-lab/LTSF-Linear/data_provider/data_loader.py#L238
        LTSF uses neither of the two approaches above. 
        Its train is strict, but val overlaps with train and test overlaps with val
        Advantage: changing history_seq_len do not affect #val_samples and #test_samples
        """
        test_ratio = 1 - train_ratio - val_ratio
        split1 = int(L * train_ratio)
        split2 = L - int(L * test_ratio)
        train_index = [
            (t - in_steps, t, t + out_steps) for t in range(in_steps, split1 - out_steps + 1)
            ]
        val_index = [
            (t - in_steps, t, t + out_steps) for t in range(split1, split2 - out_steps + 1)
            ]
        test_index = [
            (t - in_steps, t, t + out_steps) for t in range(split2, L - out_steps + 1)
            ]

    print('number of training samples: {0}'.format(len(train_index)))
    print('number of validation samples: {0}'.format(len(val_index)))
    print('number of test samples: {0}'.format(len(test_index)))


    feature_list = [data]

    if add_time_of_day:

        if file_type in ['hdf', 'csv']:
            tod = (df.index.values - df.index.values.astype('datetime64[D]')) / np.timedelta64(1, 'D')
        elif file_type == 'npz':
            tod = np.array(
                [i % steps_per_day / steps_per_day for i in range(data.shape[0])]
                )
        
        tod_tiled = np.tile(tod, [1, N, 1]).transpose((2, 1, 0))
        feature_list.append(tod_tiled)

    if add_day_of_week:

        if file_type in ['hdf', 'csv']:
            dow = df.index.dayofweek
        elif file_type == 'npz':
            dow = [
                (i // steps_per_day) % days_per_week for i in range(data.shape[0])
                ]
            
        dow_tiled = np.tile(dow, [1, N, 1]).transpose((2, 1, 0))
        feature_list.append(dow_tiled)

    processed_data = np.concatenate(feature_list, axis=-1) # (all_steps, num_nodes, num_channels+tod+dow)
    print('Data shape: {0}'.format(processed_data.shape))

    np.savez_compressed(os.path.join(output_dir, f'index_in{in_steps}_out{out_steps}.npz'), train=train_index, val=val_index, test=test_index)
    if save_data:
        np.savez_compressed(os.path.join(output_dir, f'processed_data.npz'), data=processed_data)


def main():
    
    # TODO: add save_desc() & save_graph()
    save_data = True
    data_path = os.path.join(output_dir, 'processed_data.npz')
    index_path = os.path.join(output_dir, f'index_in{regular_settings['IN_STEPS']}_out{regular_settings['OUT_STEPS']}.npz')

    if os.path.exists(data_path) and os.path.exists(index_path):
        reply = str(input(
            f"{os.path.join(output_dir, f'processed_data.npz and index_in{regular_settings['IN_STEPS']}_out{regular_settings['OUT_STEPS']}.npz')} exist. Do you want to overwrite them? (y/n) "
            )).lower().strip()
        if reply[0] != 'y':
            sys.exit(0)
    elif os.path.exists(data_path) and not os.path.exists(index_path):
        print("Generating new indices...")
        save_data = False
    
    print('Loading and preprocessing data...')
    load_and_preprocess_data(save_data)


if __name__ == '__main__':

    main()
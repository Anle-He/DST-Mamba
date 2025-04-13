import os
import sys
import yaml
import json
import argparse

import torch

sys.path.append('..')
from lib.metrics import select_loss
from lib.random import seed_everything
from lib.utils import (
    print_log,
    CustomJSONEncoder
    )

from baselines import select_model
from runners import select_runner
from data.get_dataloader import select_dataloader


if __name__ == '__main__':

    # --------- Set running env --------- #
    # TODO: Support adjust the randowm seed in scripts.
    fix_seed = 2024
    seed_everything(fix_seed)

    # Limit the number of cpu threads
    # TODO: Adjust cpu_num for long-range sequence.
    #cpu_num = 3
    #torch.set_num_threads(cpu_num) 
    #os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    #os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    #os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    #os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    #os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)    

    # Set device
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    # Set exp args
    parser = argparse.ArgumentParser()
    # Specify the model
    parser.add_argument(
        '-m',
        '--model_name',
        type=str,
        default='DSTMamba'
    )
    # Specify the task
    parser.add_argument(
        '-t',
        '--task_name',
        type=str,
        default='LTSF'
    )
    # Specify the dataset
    parser.add_argument(
        '-d',
        '--dataset_name',
        type=str,
        default='PEMS08'
    )
    # Specify .yaml config file path
    parser.add_argument(
        '-cfg',
        '--config_path',
        type=str,
        default='../baselines/DSTMamba/LTSFConfig/PEMS08.yaml'
    )
    #
    parser.add_argument(
        '-c', 
        '--compile', 
        action='store_true'
    )
    parser.add_argument(
        '-tf',
        '--test_only',
        type=bool,
        default=False
    )
    # TODO: add arg: save_resutls
    args = parser.parse_args()


    model_arch = select_model(args.model_name)
    data_path = f'../data/{args.dataset_name.upper()}'
    cfg_path = args.config_path
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)


    """
    cfg.get(key, default_value=None): no need to write in the config if not used
    cfg[key]: must be assigned in the config, else KeyError 
    """ 
    # --------- Load the model --------- #
    model = model_arch(**cfg['MODEL_PARAM']).to(DEVICE)


    # --------- Make log file --------- #
    log_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    log_path = f'../logs/{args.model_name}'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log = os.path.join(log_path, f'{args.model_name}-{args.task_name.upper()}-{args.dataset_name.upper()}-{log_time}.log')
    log = open(log, 'a')
    log.seek(0)
    log.truncate()


    # --------- Load the dataset --------- #
    print_log(f'Dataset used: {args.dataset_name.upper()}', log=log)

    (train_loader, val_loader, test_loader, SCALER) = select_dataloader(args.task_name.upper())(
        data_path,
        batch_size=cfg['GENERAL'].get('batch_size', 32),
        in_steps=cfg['DATA'].get('in_steps', 96),
        out_steps=cfg['DATA'].get('out_steps', 12),
        tod=cfg['DATA'].get('x_time_of_day'),
        dow=cfg['DATA'].get('x_day_of_week'),
        y_tod=cfg['DATA'].get('y_time_of_day'),
        y_dow=cfg['DATA'].get('y_day_of_week'),
        log=log
    )
    print_log(log=log)


    # --------- Set checkpoint path --------- #
    save_path = f'../checkpoints/{args.model_name}_{args.dataset_name.upper()}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save = os.path.join(save_path, f'{args.model_name}-{args.task_name.upper()}-{args.dataset_name.upper()}-{log_time}.pt')


    # --------- Set optim options --------- #
    criterion = select_loss(cfg['OPTIM'].get('loss', 'MSE'))(**cfg['OPTIM'].get('loss_args', {}))

    # TODO: Enable changing optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg['OPTIM'].get('initial_lr', 0.001)
    )

    lr_scheduler_type = cfg['OPTIM'].get('lr_scheduler_type', 'ExponentialLR')
    if lr_scheduler_type == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=cfg['OPTIM'].get('lr_scheduler_gamma', 0.5),
            verbose=False
        )
    elif lr_scheduler_type == 'OneCycleLR':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            steps_per_epoch=len(train_loader),
            max_lr=cfg['OPTIM'].get('initial_lr'),
            epochs=cfg['GENERAL'].get('max_epochs'),
            pct_start=cfg['OPTIM'].get('lr_scheduler_pct_start')
        )
    else: 
        raise ValueError('No such lr scheduler') 


    # --------- Set model runner --------- #
    runner = select_runner(cfg['GENERAL'].get('runner', 'LTSFRunner'))(
        cfg, device=DEVICE, scaler=SCALER, log=log
    )


    # --------- Print model args --------- #
    print_log('---------', args.model_name, '---------', log=log)
    print_log(f'Random seed = {fix_seed}', log=log)
    print_log(
        json.dumps(cfg, ensure_ascii=False, indent=4, cls=CustomJSONEncoder), log=log
    )


    # --------- Train the model --------- #
    print_log(f'Model checkpoint saved to: {save}', log=log)
    print_log(log=log)

    if not args.test_only:
        model = runner.train(
            model,
            train_loader,
            val_loader,
            optimizer,
            scheduler,
            criterion,
            max_epochs=cfg['GENERAL'].get('max_epochs', 10),
            early_stop_patience=cfg['GENERAL'].get('early_stop_patience', 3),
            compile_model=args.compile,
            verbose=1,
            save=save,
        )


    # --------- Test the model --------- #
    # TODO: Implement automatical visualization for prediction results.
    runner.test_model(model, test_loader)


    log.close()
    torch.cuda.empty_cache()
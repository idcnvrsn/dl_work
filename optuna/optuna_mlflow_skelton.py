# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 00:38:12 2020

"""

import optuna
from optuna.samplers import TPESampler
import mlflow
import argparse
from pprint import pprint
from datetime import datetime


def mlflow_callback(study, trial):
    trial_value = trial.value if trial.value is not None else float("nan")
    with mlflow.start_run(run_name=study.study_name):
        mlflow.log_params(trial.params)
        mlflow.log_metrics({"evaluation value": trial_value})


def objective(trial):
    '''
    # Categorical parameter
    optimizer = trial.suggest_categorical('optimizer', args.optimizer)

    # Int parameter
    num_layers = trial.suggest_int('num_layers', args.num_layers[0], args.num_layers[1])

    # Uniform parameter
    dropout_rate = trial.suggest_uniform('dropout_rate', args.drop_rate[0], args.drop_rate[1])

    # Loguniform parameter
    learning_rate = trial.suggest_loguniform('learning_rate', args.learning_rate[-0], args.learning_rate[1])

    # Discrete-uniform parameter
    drop_path_rate = trial.suggest_discrete_uniform('drop_path_rate', args.drop_path_rate[0], args.drop_path_rate[1], args.drop_path_rate[2])
    '''    
    # Categorical parameter
    optimizer = trial.suggest_categorical('optimizer', args.optimizer)

    # Int parameter
    num_layers = trial.suggest_categorical('num_layers', args.num_layers)

    # Uniform parameter
    dropout_rate = trial.suggest_categorical('dropout_rate', args.dropout_rate)

    return 1.0


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='このプログラムの説明', fromfile_prefix_chars='@')
    
#    parser.add_argument('filename', help='ファイル名')
#    parser.add_argument('input_dir', help='入力ディレクトリ')
    parser.add_argument('-sl', '--sampler', default="grid", choices=['grid', 'random', 'tpe'], help='samplerを指定する')
    parser.add_argument('-tr', '--n_trials', type=int, default=20, help='最適化トライアル数')
    parser.add_argument('-to', '--timeout', type=int, default=600, help='最適化タイムアウト時間')
    parser.add_argument('-exp', '--experiment', default="Default", help='実験名')

    """
    parser.add_argument('--arg3')
    parser.add_argument('-a', '--arg4')
    """
    parser.add_argument('-o', '--optimizer',nargs="*", default=['MomentumSGD', 'Adam'], help='')
    parser.add_argument('-n', '--num_layers',nargs="*", type=int, default=[1,3], help='')
    parser.add_argument('-d', '--dropout_rate',nargs="*", type=float, default=[0.0, 1.0], help='')
    parser.add_argument('-l', '--learning_rate',nargs="*", type=float, default=[1e-5, 1e-2], help='')
    parser.add_argument('-dr', '--drop_path_rate',nargs="*", type=float, default=[0.0, 1.0, 0.1], help='')
#    parser.add_argument('-o', '--optimizer',nargs="*", type=float, default=['a','b','c'], help='')
    
    args = parser.parse_args()
    pprint(args.__dict__)
    
    if args.sampler is "grid":
        search_space = {
        'optimizer' : args.optimizer,
        'num_layers': args.num_layers,
        'dropout_rate': args.dropout_rate
        }
        sampler=optuna.samplers.GridSampler(search_space)
        n_trials=1
        for value in search_space.values():
            n_trials*=len(value)
    elif args.sampler is "random":
        sampler=optuna.samplers.RandomSampler()
        n_trials=args.n_trials
    else:
        sampler=TPESampler(**TPESampler.hyperopt_parameters())
        n_trials=args.n_trials

    print("n_trials:", n_trials)

    if n_trials == 1:
        mlflow.set_experiment(args.experiment)
    else:
        mlflow.set_experiment(args.experiment+"_"+datetime.now().strftime('%Y%m%d_%H:%M:%S'))

    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=n_trials, timeout=args.timeout, callbacks=[mlflow_callback])

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

# -*- coding: utf-8 -*-
"""
このスケルトンをベースに訓練スクリプトを作成するとoptunaとmlflowの機能を利用できる.
パラメータの指定は下記4種類
・シングルパラメータ
・grid search
・random search
・tpe search
"""

import optuna
from optuna.samplers import GridSampler, RandomSampler, TPESampler
import mlflow
import argparse
from pprint import pprint
from datetime import datetime
import random
import numpy as np
import copy

class OptunaMlFlow:
    def __init__(self, argument, grid_search_space=None):
        self.name = ''
        self.argument=argument
        self.grid_search_space=grid_search_space

        if self.argument.sampler == "grid":
            assert self.grid_search_space is not None, "grid search spaceを指定してください"

            self.sampler=GridSampler(self.grid_search_space)
            self.n_trials=1
            for value in self.grid_search_space.values():
                self.n_trials*=len(value)

            # トライアル回数制限
#            if self.n_trials > self.argument.n_trials:
#                self.n_trials = self.argument.n_trials

            self.obj_func_name = self.objective_grid
        elif self.argument.sampler == "random":
            self.sampler=RandomSampler(seed=self.argument.seed)
            self.n_trials=self.argument.n_trials
            self.obj_func_name = self.objective_no_grid
        else:
            self.sampler=TPESampler(**TPESampler.hyperopt_parameters(), seed=self.argument.seed)
            self.n_trials=self.argument.n_trials
            self.obj_func_name = self.objective_no_grid

        if self.n_trials == 1:
            try:
                mlflow.set_experiment(self.argument.experiment)
            except Exception as e:
                print(e)
        else:
            try:
                mlflow.set_experiment(self.argument.experiment+"_"+datetime.now().strftime('%Y%m%d_%H:%M:%S'))
            except Exception as e:
                print(e)

        self.study = optuna.create_study(sampler=self.sampler)

    def start_run(self, trial):
        try:
            mlflow.start_run(run_name='trial_'+'{:0006}'.format(trial.number))
        except Exception as e:
            print(e)

    def end_run(self):
        try:
            mlflow.end_run()
        except Exception as e:
            print(e)

    def mlflow_callback(self, study, trial):
        trial_value = trial.value if trial.value is not None else float("nan")
        with mlflow.start_run(run_name=study.study_name):
            mlflow.log_params(trial.params)
            mlflow.log_metrics({"evaluation value": trial_value})

    def add_dict_key_prefix(self, prefix, _dict):
        new_dict={}
        for k,v in _dict.items():
            new_dict[prefix+k]=v
        return new_dict

    def add_dict_key_postfix(self, _dict, postfix):
        new_dict={}
        for k,v in _dict.items():
            new_dict[k+postfix]=v
        return new_dict

    # mlflowにロギング
    def log_trial(self, trial):
        mlflow.log_params(self.add_dict_key_prefix("args_", self.argument.__dict__, ))
        mlflow.log_param("n_trials", self.n_trials)
        mlflow.log_params(self.add_dict_key_postfix(trial.params, "_trial_params"))

    # 最適化したい処理を記述する(一つのrunに対応する)
    def trial_process(self, trial, optimizer, num_layers, dropout_rate):
        # mlflowのrunを開く
        self.start_run(trial)

        # mlflowにtrialごとの情報をロギング
        self.log_trial(trial)

        mlflow.log_metric("score", 1.0, 0)

        # mlflowのrunを閉じる
        self.end_run()

        return 1.0

    # ランダムおよびTPEサーチを行うための目的関数
    def objective_no_grid(self, trial):
        '''
        # Categorical parameter
        optimizer = trial.suggest_categorical('optimizer', self.argument.optimizer)

        # Int parameter
        num_layers = trial.suggest_int('num_layers', self.argument.num_layers[0], self.argument.num_layers[1])

        # Uniform parameter
        dropout_rate = trial.suggest_uniform('dropout_rate', self.argument.dropout_rate[0], self.argument.dropout_rate[1])

        # Loguniform parameter
        learning_rate = trial.suggest_loguniform('learning_rate', self.argument.learning_rate[0], self.argument.learning_rate[1])

        # Discrete-uniform parameter
        drop_path_rate = trial.suggest_discrete_uniform('drop_path_rate', self.argument.drop_path_rate[0], self.argument.drop_path_rate[1], self.argument.drop_path_rate[2])
        '''

        # Categorical parameter
        optimizer = trial.suggest_categorical('optimizer', self.argument.optimizer)

        # Int parameter
        num_layers = trial.suggest_int('num_layers', self.argument.num_layers[0], self.argument.num_layers[1])

        # Uniform parameter
        dropout_rate = trial.suggest_uniform('dropout_rate', self.argument.dropout_rate[0], self.argument.dropout_rate[1])

        """
        # このイテレーションで使うパラメータの組み合わせを構築する
        _args = copy.deepcopy(self.argument)
        _args.lr=_lr
        _args.batch_size=_batch_size
        _args.loss_type=_loss_type
        """

        # ここで一つのパラメータの組み合わせ(trial)について評価する
        result=self.trial_process(trial, optimizer, num_layers, dropout_rate)

        return result

    # グリッドサーチおよび単一のパラメータでの学習を行う際の目的関数
    def objective_grid(self, trial):
        '''
        パラメータは原則trial,suggest_categorical()で指定する。 
        
        optimizer = trial.suggest_categorical('optimizer', self.argument.optimizer)

        num_layers = trial.suggest_categorical('num_layers', self.argument.num_layers)

        dropout_rate = trial.suggest_categorical('dropout_rate', self.argument.dropout_rate)

        learning_rate = trial.suggest_categorical('learning_rate', self.argument.learning_rate)

        drop_path_rate = trial.suggest_categorical('drop_path_rate', self.argument.drop_path_rate)
        '''    

        # Categorical parameter
        optimizer = trial.suggest_categorical('optimizer', self.argument.optimizer)

        # Int parameter
        num_layers = trial.suggest_categorical('num_layers', self.argument.num_layers)

        # Uniform parameter
        dropout_rate = trial.suggest_categorical('dropout_rate', self.argument.dropout_rate)

        """
        # このイテレーションで使うパラメータの組み合わせを構築する
        _args = copy.deepcopy(args)
        _args.lr=_lr
        _args.batch_size=_batch_size
        _args.loss_type=_loss_type
        """
        
        # ここで一つのパラメータの組み合わせ(trial)について評価する
        result=self.trial_process(trial, optimizer, num_layers, dropout_rate)

        return result

    def optimize(self):
        self.study.optimize(self.obj_func_name, n_trials=self.n_trials, timeout=self.argument.timeout)#, callbacks=[self.mlflow_callback])

    def get_result_text(self):
        text=str()
        text+="Number of finished trials: {}".format(len(self.study.trials))+"\n"

        text+="Best trial:"+"\n"
        trial = self.study.best_trial

        text+="  Value: {}".format(trial.value)+"\n"

        text+="  Params: "+"\n"
        for key, value in trial.params.items():
            text+="    {}: {}".format(key, value)+"\n"

        return text

def main():
    parser = argparse.ArgumentParser(description='optunaとmlflowを利用した学習スクリプト', formatter_class=argparse.ArgumentDefaultsHelpFormatter, fromfile_prefix_chars='@')
    
    # optunaとmlflowに設定するオプション
    parser.add_argument('-sl', '--sampler', default="grid", choices=['grid', 'random', 'tpe'], help='samplerを指定する(シングルパラメータで学習する場合はgridを指定する)')
    parser.add_argument('-tr', '--n_trials', type=int, default=25, help='最適化トライアル数(シングルパラメータ、グリッドサーチ時は無視される)')
    parser.add_argument('-to', '--timeout', type=int, default=None, help='最適化タイムアウト時間')
    parser.add_argument('-exp', '--experiment', default="Default", help='実験名。1通りのパラメータセットで学習する際には設定した実験名となる。複数のパラメータセットを探索する際には日時を付与して個別の実験名とする。')
    parser.add_argument('-bs', '--batch_size', type=int, default=3, help='バッチサイズ')
    parser.add_argument('--seed', type=int,default=4321, help='random seed')

    # ここでチューニングしたいパラメータを設定する
    # optunaの探索空間を設定する　gridサーチの場合ここで設定した組みわせ全てを探索する。
    parser.add_argument('-o', '--optimizer',nargs="*", default=['MomentumSGD', 'Adam'], help='')
    parser.add_argument('-n', '--num_layers',nargs="*", type=int, default=[1,3], help='')
    parser.add_argument('-d', '--dropout_rate',nargs="*", type=float, default=[0.0, 1.0], help='')
    parser.add_argument('-l', '--learning_rate',nargs="*", type=float, default=[1e-5, 1e-2], help='')
    parser.add_argument('-dr', '--drop_path_rate',nargs="*", type=float, default=[0.0, 1.0, 0.1], help='')
#    parser.add_argument('-o', '--optimizer',nargs="*", type=float, default=['a','b','c'], help='')
    
    args = parser.parse_args()
    pprint(args.__dict__)

    # グリッドサーチする場合はここでチューニングしたいパラメータ空間を定義してコンストラクタに渡す。
    # さらに目的関数内でも組み合わせを取り出す処理を記述する
    grid_search_space = {
        'optimizer' : args.optimizer,
        'num_layers': args.num_layers,
        'dropout_rate': args.dropout_rate
    }

    optuna_mlflow=OptunaMlFlow(args, grid_search_space)
    optuna_mlflow.optimize()

    print("n_trials:", optuna_mlflow.n_trials)

    print(optuna_mlflow.get_result_text())

if __name__ == '__main__':
    main()

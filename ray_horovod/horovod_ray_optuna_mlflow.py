import torch
import numpy as np
import time
from base_mlflow import BaseMlFlow

import ray
from ray.tune.integration.horovod import DistributedTrainableCreator
from ray.tune.suggest.optuna import OptunaSearch
from ray.tune.suggest import ConcurrencyLimiter

def sq(x):
    m2 = 1.
    m1 = -20.
    m0 = 50.
    return m2 * x * x + m1 * x + m0


def qu(x):
    m3 = 10.
    m2 = 5.
    m1 = -20.
    m0 = -5.
    return m3 * x * x * x + m2 * x * x + m1 * x + m0


class Net(torch.nn.Module):
    def __init__(self, mode="sq"):
        super(Net, self).__init__()

        if mode == "square":
            self.mode = 0
            self.param = torch.nn.Parameter(torch.FloatTensor([1., -1.]))
        else:
            self.mode = 1
            self.param = torch.nn.Parameter(torch.FloatTensor([1., -1., 1.]))

    def forward(self, x):
        if ~self.mode:
            return x * x + self.param[0] * x + self.param[1]
        else:
            return_val = 10 * x * x * x
            return_val += self.param[0] * x * x
            return_val += self.param[1] * x + self.param[2]
            return return_val

class HorovodRayOptunaMlflow(BaseMlFlow):
    def __init__(self, args):
        super().__init__(args)
        self.args=args

    def trial_process(self, config):
        import torch
        import horovod.torch as hvd

        hvd.init()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mode = config["mode"]
        net = Net(mode).to(device)
        optimizer = torch.optim.SGD(
            net.parameters(),
            lr=config["lr"],
        )
        optimizer = hvd.DistributedOptimizer(optimizer)

        num_steps = 5
        print(hvd.size())
        np.random.seed(1 + hvd.rank())
        torch.manual_seed(1234)
        # To ensure consistent initialization across slots,
        hvd.broadcast_parameters(net.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)

        if hvd.rank() == 0:
            print("hvd.rank():",hvd.rank())
            self.start_run()

        start = time.time()
        x_max = config["x_max"]
        for step in range(1, num_steps + 1):
            features = torch.Tensor(np.random.rand(1) * 2 * x_max -
                                    x_max).to(device)
            if mode == "square":
                labels = sq(features)
            else:
                labels = qu(features)
            optimizer.zero_grad()
            outputs = net(features)
            loss = torch.nn.MSELoss()(outputs, labels)
            loss.backward()

            optimizer.step()
            time.sleep(0.1)
            ray.tune.report(loss=loss.item())
        total = time.time() - start
        print(f"Took {total:0.3f} s. Avg: {total / num_steps:0.3f} s.")
    
        if hvd.rank() == 0:
            self.end_run()

    def tune(self,
            hosts_per_trial,
            slots_per_host,
            num_samples,
            use_gpu,
            mode="square",
            x_max=1.):
        # search algorithm
        search_alg = OptunaSearch(metric='loss', mode='min')
        search_alg = ConcurrencyLimiter(search_alg, max_concurrent=4)

        horovod_trainable = DistributedTrainableCreator(
            self.trial_process,
            use_gpu=use_gpu,
            num_hosts=hosts_per_trial,
            num_slots=slots_per_host,
            replicate_pem=False)

        analysis = ray.tune.run(
            horovod_trainable,
            metric="loss",
            mode="min",
            config=self.args.search_space,
            local_dir=".",
            num_samples=num_samples,
            fail_fast=True,
            search_alg=search_alg)
        print("Best hyperparameters found were: ", analysis.best_config)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", type=str, default="square", choices=["square", "cubic"])
    parser.add_argument(
        "--learning_rate", type=float, default=0.1, dest="learning_rate")
    parser.add_argument("--x_max", type=float, default=1., dest="x_max")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=("Finish quickly for testing."))
    parser.add_argument("--hosts-per-trial", type=int, default=1)
    parser.add_argument("--slots-per-host", type=int, default=3)
    parser.add_argument(
        "--server-address",
        type=str,
        default=None,
        required=False,
        help="The address of server to connect to if using "
        "Ray Client.")

    parser.add_argument('-exp', '--experiment', default="Default", help='実験名。1通りのパラメータセットで学習する際には設定した実験名となる。複数のパラメータセットを探索する際には日時を付与して個別の実験名とする。')
    parser.add_argument('-bs', '--batch_size', type=int, default=3, help='バッチサイズ')
    parser.add_argument('--seed', type=int,default=4321, help='random seed')

    """
    # ここでチューニングしたいパラメータを設定する
    # optunaの探索空間を設定する　gridサーチの場合ここで設定した組みわせ全てを探索する。
    parser.add_argument('-o', '--optimizer',nargs="*", default=['MomentumSGD', 'Adam'], help='')
    parser.add_argument('-n', '--num_layers',nargs="*", type=int, default=[1,3], help='')
    parser.add_argument('-d', '--dropout_rate',nargs="*", type=float, default=[0.0, 1.0], help='')
    parser.add_argument('-l', '--learning_rate',nargs="*", type=float, default=[1e-5, 1e-2], help='')
    parser.add_argument('-dr', '--drop_path_rate',nargs="*", type=float, default=[0.0, 1.0, 0.1], help='')
#    parser.add_argument('-o', '--optimizer',nargs="*", type=float, default=['a','b','c'], help='')
    """

    args, _ = parser.parse_known_args()

    if args.smoke_test:
        ray.init(num_cpus=2)
    elif args.server_address:
        ray.util.connect(args.server_address)

    # import ray
    # ray.init(address="auto")  # assumes ray is started with ray up

    args.search_space={
                "lr": ray.tune.uniform(0.1, 1),
                "mode": args.mode,
                "x_max": args.x_max
            }

    runner=HorovodRayOptunaMlflow(args)
    runner.tune(
        hosts_per_trial=args.hosts_per_trial,
        slots_per_host=args.slots_per_host,
        num_samples=2 if args.smoke_test else 10,
        use_gpu=args.gpu,
        mode=args.mode,
        x_max=args.x_max)

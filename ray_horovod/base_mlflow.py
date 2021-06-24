import mlflow

class BaseMlFlow:
    def __init__(self, args):
        self.args=args
#        self.experiment_id, self.run_id=self.parse_experiment_run_id(self.args.experiment_run)

#        self.start_run()

#    def __del__(self):
#        self.end_run()

    def parse_experiment_run_id(self, experiment_id):
        return experiment_id.split("/")

    def start_run(self):
        try:
            mlflow.start_run()

#            experiment = mlflow.get_experiment()
#            self.artifact_location=experiment.artifact_location
#            print("Artifact Location: {}".format(self.artifact_location))
#            print("Artifact Path: {}".format(self.get_artifacts_path()))  
#            print("Artifact uri: {}".format(mlflow.get_artifact_uri()))

        except Exception as e:
            print(e)

    def end_run(self):
        try:
            mlflow.end_run()
            print("base finished.")
        except Exception as e:
            print(e)

    def get_artifacts_path(self):
        path=self.artifact_location.split("file:/")[1]
        return path+os.sep+self.run_id+os.sep+"artifacts"

    # ここに処理を記載する
    def run(self):
        pass

def main():
    import argparse
    from pprint import pprint
    import os
    parser = argparse.ArgumentParser(description='このプログラムの説明', formatter_class=argparse.ArgumentDefaultsHelpFormatter, fromfile_prefix_chars='@')

    args = parser.parse_args()
    pprint(args.__dict__)

    base=BaseMlFlow(args)
    base.start_run()
    base.end_run()

if __name__ == '__main__':
    main()

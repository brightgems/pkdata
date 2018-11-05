import click
from src.pipeline_manager import PipelineManager
import warnings

warnings.filterwarnings(action='ignore')
pipeline_manager = PipelineManager()


@click.group()
def main():
    pass


@main.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-d', '--dev_mode', help='if true only a small sample of data will be used', required=False)
def train(pipeline_name, dev_mode):
    pipeline_manager.train_evaluate(pipeline_name, dev_mode)


@main.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-d', '--dev_mode', help='if true only a small sample of data will be used', required=False)
def predict(pipeline_name, dev_mode):
    pipeline_manager.predict(pipeline_name, dev_mode)


@main.command()
@click.option('-s', '--dataset', help='dataset to be processed', required=True)
@click.option('-d', '--dev_mode', help='if true only a small sample of data will be used', required=False)
def preprocess(dataset, dev_mode):
    pipeline_manager.prepare_dataset(dataset, dev_mode)


if __name__ == "__main__":
    main()
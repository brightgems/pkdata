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
    pipeline_manager.train(pipeline_name, dev_mode)


@main.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-d', '--dev_mode', help='if true only a small sample of data will be used', required=False)
def evaluate(pipeline_name, dev_mode):
    pipeline_manager.evaluate(pipeline_name, dev_mode)


@main.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-s', '--submit_predictions', help='submit predictions if true', type=bool, required=False)
@click.option('-d', '--dev_mode', help='if true only a small sample of data will be used', required=False)
def predict(pipeline_name, dev_mode, submit_predictions):
    import pdb; pdb.set_trace()
    pipeline_manager.predict(pipeline_name, dev_mode, submit_predictions)


@main.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-s', '--submit_predictions', help='submit predictions if true', type=bool, required=False)
@click.option('-d', '--dev_mode', help='if true only a small sample of data will be used', required=False)
def train_evaluate_predict(pipeline_name, submit_predictions, dev_mode):
    pipeline_manager.train(pipeline_name, dev_mode)
    pipeline_manager.evaluate(pipeline_name, dev_mode)
    pipeline_manager.predict(pipeline_name, dev_mode, submit_predictions)


@main.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-d', '--dev_mode', help='if true only a small sample of data will be used', required=False)
def train_evaluate(pipeline_name, dev_mode):
    pipeline_manager.train(pipeline_name, dev_mode)
    pipeline_manager.evaluate(pipeline_name, dev_mode)


@main.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-s', '--submit_predictions', help='submit predictions if true', type=bool, required=False)
@click.option('-d', '--dev_mode', help='if true only a small sample of data will be used', required=False)
def evaluate_predict(pipeline_name, submit_predictions, dev_mode):
    pipeline_manager.evaluate(pipeline_name, dev_mode)
    pipeline_manager.predict(pipeline_name, dev_mode, submit_predictions)


@main.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-d', '--dev_mode', help='if true only a small sample of data will be used', required=False)
def train_evaluate_cv(pipeline_name, dev_mode):
    pipeline_manager.train_evaluate_cv(pipeline_name, dev_mode)


@main.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-s', '--submit_predictions', help='submit predictions if true', type=bool, required=False)
@click.option('-d', '--dev_mode', help='if true only a small sample of data will be used', required=False)
def train_evaluate_predict_cv(pipeline_name, submit_predictions, dev_mode):
    pipeline_manager.train_evaluate_predict_cv(pipeline_name, dev_mode, submit_predictions)


@main.command()
def submit():
    pipeline_manager.submit_result()

if __name__ == "__main__":
    main()
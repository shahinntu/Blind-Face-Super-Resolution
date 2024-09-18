import sys

from src import parse_args, MLPipeline

args = parse_args(sys.argv[1:])

train_pipeline = MLPipeline(args)
train_pipeline.run()

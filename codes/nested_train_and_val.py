import yaml
from utils import TrainModel, EvaluatingConfidenceScores
import csv
import argparse
import os
from ultralytics import YOLO

#Argument parser
parser = argparse.ArgumentParser(prog="MBGd with YOLO",description="MBG Eval")
parser.add_argument("--config-file", default=None, help="path to config.yaml")
parser.add_argument("--fold-for-test", default=None, help="outer loop test fold number")
parser.add_argument("--fold-for-validation", default=None, help="inner loop validation fold number")
parser.add_argument("--object", default=None, help="object to detect")
args = parser.parse_args()
#set the dataset and config file paths
config_file = args.config_file
fold_test = args.fold_for_test
fold_val = args.fold_for_validation
obj = args.object

# Load the general YAML configuration
with open(config_file, 'r') as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)

#get arguments from config file
experiment_name = config['EXPERIMENT_NAME']
pre_trained_model = config['PRE_TRAINED_MODEL']
path_hparams = config ['PARAMS']
epochs = config['EPOCHS']
patience = config['PATIENCE']
output_dir = config['OUTPUT_DIR']
dataset = config['DATASET'][obj.upper()][f'TEST_FOLD{fold_test}'][f'VAL_FOLD{fold_val}']

#dataset for simple tests
#uncomment line below to use this simple and small dataset
#dataset = config['DATASET']["TIRE"]['teste'] #remove when tests are finished

#----------------------------------------------------------------------------------------------------
#training model
model = YOLO(pre_trained_model)
result_path_train = f'{experiment_name}/outer_test_fold{fold_test}/inner_val_fold{fold_val}/train'

results_train = TrainModel(model=model,
                           dataset=dataset,
                           experiment_name=result_path_train,
                           hyp_params=path_hparams,
                           epochs=epochs,
                           patience=patience,
                           output_dir=output_dir)

#----------------------------------------------------------------------------------------------------
#search for best confidence score
model_to_evaluate = YOLO(str(results_train.save_dir) + '/weights/best.pt')
result_path_val = f'{experiment_name}/outer_test_fold{fold_test}/inner_val_fold{fold_val}/val/iter'

results_data = EvaluatingConfidenceScores(model=model_to_evaluate,
                                          dataset=dataset,
                                          output_dir=output_dir,
                                          experiment_name=result_path_val)

#----------------------------------------------------------------------------------------------------
#Saving result metrics
metrics_file_path = f'{output_dir}/{experiment_name}/outer_test_fold{fold_test}/inner_val_fold{fold_val}/val_metrics.csv'
output_directory = os.path.dirname(metrics_file_path)
os.makedirs(output_directory, exist_ok=True)

with open(metrics_file_path, mode='a', newline='') as file:
    # Create a csv.writer object
    writer = csv.writer(file)
    for line in results_data:
        writer.writerow(line)
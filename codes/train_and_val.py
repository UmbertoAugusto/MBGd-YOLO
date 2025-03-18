import yaml
from utils import *
import csv
import argparse
import os

#Argument parser
parser = argparse.ArgumentParser(prog="MBGd with YOLO",description="MBG Eval")
parser.add_argument("--config-file", default=None, help="path to config.yaml")
parser.add_argument("--object", default=None, help="object to detect")
parser.add_argument("--fold", default=None, help="current fold number")
args = parser.parse_args()
#set the dataset and config file paths
config_file = args.config_file
fold_number = args.fold
obj = args.object

# Load the general YAML configuration
with open(config_file, 'r') as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)

pre_trained_model = config['PRE_TRAINED_MODEL']
experiment_name = config['EXPERIMENT_NAME']
epochs = config['EPOCHS']
img_size = config['IMAGE_SIZE']
patience = config['PATIENCE']
output_dir = config['OUTPUT_DIR']
num_folds = config['NUM_FOLDS']
dataset= config['DATASET'][obj.upper()][f'FOLD{fold_number}']

#dataset for simple tests
dataset_teste = config['DATASET']["TIRE"]['teste']

best_F1_scores = {}
best_conf_scores = {}
result_data = [['fold','score','F1']]

#train model
model = YOLO(pre_trained_model)
#model.to('cuda')
result_path_train = experiment_name+'/fold'+str(fold_number)+'/train'
results_train = TrainModel(model=model,
                           dataset=dataset_teste, #change for dataset when finish testing code
                           experiment_name=result_path_train,
                           epochs=epochs,
                           patience=patience,
                           img_size=img_size,
                           output_dir=output_dir)

#search for best confidence score
model_to_evaluate = YOLO(str(results_train.save_dir) + '/weights/best.pt')
#model_to_evaluate.to('cuda')
result_path_val = experiment_name+'/fold'+str(fold_number)+'/val/iter'
best_conf_score, best_F1_score = ConfidenceThresholdOptimization(model=model_to_evaluate,
                                                                 dataset=dataset_teste, #change for dataset when finish testing code
                                                                 output_dir=output_dir,
                                                                 experiment_name=result_path_val)

#Saving result metrics
result_data.append([fold_number,best_conf_score,best_F1_score])
#Writing file
metrics_file_path = output_dir+'/'+experiment_name+'/fold'+str(fold_number)+'/metrics.txt'
with open(metrics_file_path, mode='w') as file:
    file.write(f"Fold {fold_number}\nscore: {best_conf_score}\nF1: {best_F1_score}")

'''#Saving metrics of all folds
csv_file_path = output_dir+'/'+experiment_name+'/metrics.csv'
# Open the file in write mode
with open(csv_file_path, mode='w', newline='') as file:
    # Create a csv.writer object
    writer = csv.writer(file)
    # Write data to the CSV file
    writer.writerows(result_data)'''
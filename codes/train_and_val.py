import yaml
from utils import TrainModel, ConfidenceThresholdOptimization
import csv
import argparse
import os
from ultralytics import YOLO

#Argument parser
parser = argparse.ArgumentParser(prog="MBGd with YOLO",description="MBG Eval")
parser.add_argument("--config-file", default=None, help="path to config.yaml")
parser.add_argument("--fold", default=None, help="current fold number")
parser.add_argument("--object", default=None, help="object to detect")
args = parser.parse_args()
#set the dataset and config file paths
config_file = args.config_file
fold_number = args.fold
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
dataset= config['DATASET'][obj.upper()][f'FOLD{fold_number}']

#dataset for simple tests
#uncomment line below to use this simple and small dataset
#dataset = config['DATASET']["TIRE"]['teste'] #remove when tests are finished

result_data = [['fold','score','F1']] #header for .csv file with results

#train model
model = YOLO(pre_trained_model)

result_path_train = experiment_name+'/fold'+str(fold_number)+'/train'
results_train = TrainModel(model=model,
                           dataset=dataset,
                           experiment_name=result_path_train,
                           hyp_params=path_hparams,
                           epochs=epochs,
                           patience=patience,
                           output_dir=output_dir)

#search for best confidence score
model_to_evaluate = YOLO(str(results_train.save_dir) + '/weights/best.pt')

result_path_val = experiment_name+'/fold'+str(fold_number)+'/val/iter'
best_conf_score, best_F1_score = ConfidenceThresholdOptimization(model=model_to_evaluate,
                                                                 dataset=dataset,
                                                                 output_dir=output_dir,
                                                                 experiment_name=result_path_val)

#Saving result metrics
result_data.append([fold_number,best_conf_score,best_F1_score])
#Writing file
metrics_file_path = output_dir+'/'+experiment_name+'/fold'+str(fold_number)+'/metrics.txt'
with open(metrics_file_path, mode='w') as file:
    file.write(f"Fold {fold_number}\nscore: {best_conf_score}\nF1: {best_F1_score}")


#Saving metrics of all folds
csv_file_path = output_dir+'/'+experiment_name+'/metrics.csv'

write_titles = False #indica se precisa escrever nome das colunas ainda
if not os.path.exists(csv_file_path):
    write_titles = True 

# Open the file
with open(csv_file_path, mode='a', newline='') as file:
    # Create a csv.writer object
    writer = csv.writer(file)
    # Write data to the CSV file
    if write_titles:
        writer.writerow(result_data[0])
    writer.writerow(result_data[1])
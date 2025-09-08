import yaml
from utils import TrainModel, TestModel, GetBestConfScore
import csv
import argparse
import os
from ultralytics import YOLO

#Argument parser
parser = argparse.ArgumentParser(prog="MBGd with YOLO",description="MBG Eval")
parser.add_argument("--config-file", default=None, help="path to config.yaml")
parser.add_argument("--fold-test", default=None, help="current fold number")
parser.add_argument("--object", default=None, help="object to detect")
args = parser.parse_args()
#set the dataset and config file paths
config_file = args.config_file
fold_test = args.fold
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
dataset = config['DATASET'][obj.upper()][f'TEST_FOLD{fold_test}']['FINAL_TRAIN_TEST']

#----------------------------------------------------------------------------------------------------
#training model
model = YOLO(pre_trained_model)
result_path_train = experiment_name+'/fold'+str(fold_test)+'/train'

results_train = TrainModel(model=model,
                           dataset=dataset,
                           experiment_name=result_path_train,
                           hyp_params=path_hparams,
                           epochs=epochs,
                           patience=patience,
                           output_dir=output_dir)

#----------------------------------------------------------------------------------------------------
#evaluating model
model_to_evaluate = YOLO(str(results_train.save_dir) + '/weights/best.pt')
result_path = experiment_name+'/fold'+str(fold_test)+'/test'

#getting best conf_score
conf_score = GetBestConfScore(outer_loop_test_fold=fold_test,
                              output_dir=output_dir,
                              experiment_name=experiment_name)

results_test = TestModel (model=model_to_evaluate,
                          dataset=dataset,
                          output_dir=output_dir,
                          experiment_name=result_path,
                          conf_score = conf_score)

TruePositives = results_test[0]
FalsePositives = results_test[1]
FalseNegatives = results_test[2]
Precision = results_test[3]
Recall = results_test[4]
F1 = results_test[5]
mAP50_95 = results_test[6]
mAP50 = results_test[7]

#----------------------------------------------------------------------------------------------------
#Saving result metrics
result_data = [['fold','score','TP','FP','FN','Precision','Recall','F1','mAP@50-95','mAP@50']] #header for .csv file with results
result_data.append([fold_test,conf_score,TruePositives,FalsePositives,FalseNegatives,Precision,Recall,F1,mAP50_95,mAP50])

#Saving results of all folds together
csv_file_path = f'{output_dir}/{experiment_name}/test_results.csv'

write_titles = False #indicates if it is needed to write csv file header
if not os.path.exists(csv_file_path):
    os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
    write_titles = True 

# Open the file
with open(csv_file_path, mode='a', newline='') as file:
    # Create a csv.writer object
    writer = csv.writer(file)
    # Write data to the CSV file
    if write_titles:
        writer.writerow(result_data[0])
    writer.writerow(result_data[1])
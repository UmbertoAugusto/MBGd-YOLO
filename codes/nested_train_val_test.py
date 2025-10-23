import yaml
from utils import TrainModel, ConfidenceThresholdOptimization, TestModel
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
parser.add_argument("--database", default=None, help="database to use: integer or tiled")
args = parser.parse_args()
#set the dataset and config file paths
config_file = args.config_file
fold_test = args.fold_for_test
fold_val = args.fold_for_validation
obj = args.object
database = args.database

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
dataset = config['DATASET'][database.upper()][obj.upper()][f'TEST_FOLD{fold_test}'][f'VAL_FOLD{fold_val}']
original_annotations_val_json_path = ''
original_annotations_test_json_path = ''
if database.lower() == "tiled":
    original_annotations_val_json_path = config['DATASET'][database.upper()][obj.upper()][f'ORIGINAL_JSON_FOLD{fold_val}']
    original_annotations_test_json_path = config['DATASET'][database.upper()][obj.upper()][f'ORIGINAL_JSON_FOLD{fold_test}']

#dataset for simple tests
#uncomment line below to use this simple and small dataset
#dataset = config['DATASET']["TIRE"]['teste'] #remove when tests are finished

#----------------------------------------------------------------------------------------------------
#training model
model = YOLO(pre_trained_model)
result_path_train = f'{experiment_name}/test_fold{fold_test}/val_fold{fold_val}/train'

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
result_path_val = f'{experiment_name}/test_fold{fold_test}/val_fold{fold_val}/val/iter'

best_conf, best_F1 = ConfidenceThresholdOptimization(model=model_to_evaluate,
                                                     dataset=dataset,
                                                     output_dir=output_dir,
                                                     experiment_name=result_path_val,
                                                     database=database,
                                                     annotations_json_path=original_annotations_val_json_path)

#----------------------------------------------------------------------------------------------------
#Saving validation metrics
result_data = [['test_fold','val_fold','score','F1']] #header for .csv file
result_data.append([fold_test,fold_val,best_conf,best_F1])
#Writing file
val_metrics_file_path = f'{output_dir}/{experiment_name}/val_results.csv'

write_titles = False #indicates if it is needed to write csv file header
if not os.path.exists(val_metrics_file_path):
    write_titles = True
    os.makedirs(os.path.dirname(val_metrics_file_path), exist_ok=True)

# Open the file
with open(val_metrics_file_path, mode='a', newline='') as file:
    # Create a csv.writer object
    writer = csv.writer(file)
    # Write data to the CSV file
    if write_titles:
        writer.writerow(result_data[0])
    writer.writerow(result_data[1])

#----------------------------------------------------------------------------------------------------
#evaluating model
model_to_evaluate = YOLO(str(results_train.save_dir) + '/weights/best.pt')
result_path = f'{experiment_name}/test_fold{fold_test}/val_fold{fold_val}/test'

results_test = TestModel(model=model_to_evaluate,
                          dataset=dataset,
                          output_dir=output_dir,
                          experiment_name=result_path,
                          conf_score = best_conf,
                          database = database,
                          annotations_json_path=original_annotations_test_json_path)

TruePositives = results_test[0]
FalsePositives = results_test[1]
FalseNegatives = results_test[2]
Precision = results_test[3]
Recall = results_test[4]
F1 = results_test[5]

#----------------------------------------------------------------------------------------------------
#Saving test metrics
result_data = [['fold_test','fold_val','conf_score','TP','FP','FN','Precision','Recall','F1','mAP@50-95','mAP@50']] #header for .csv file with results
result_data.append([fold_test,fold_val,best_conf,TruePositives,FalsePositives,FalseNegatives,Precision,Recall,F1])

#Saving results of all folds together
test_metrics_file_path = f'{output_dir}/{experiment_name}/test_results.csv'

write_titles = False #indicates if it is needed to write csv file header
if not os.path.exists(test_metrics_file_path):
    os.makedirs(os.path.dirname(test_metrics_file_path), exist_ok=True)
    write_titles = True 

# Open the file
with open(test_metrics_file_path, mode='a', newline='') as file:
    # Create a csv.writer object
    writer = csv.writer(file)
    # Write data to the CSV file
    if write_titles:
        writer.writerow(result_data[0])
    writer.writerow(result_data[1])
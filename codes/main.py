import yaml
from utils import *
import csv

print(" --- Welcome to MBGd workflow ! ---")

# Load the general YAML configuration
with open("/home/umberto.pereira/Mosquitoes/YOLO/configs/config.yaml", 'r') as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)

pre_trained_model = config['PRE_TRAINED_MODEL']
experiment_name = config['EXPERIMENT_NAME']
epochs = config['EPOCHS']
img_size = config['IMAGE_SIZE']
patience = config['PATIENCE']
dataset_fold1 = config['DATASET']['fold1']
dataset_fold2 = config['DATASET']['fold2']
dataset_fold3 = config['DATASET']['fold3']
dataset_fold4 = config['DATASET']['fold4']
dataset_fold5 = config['DATASET']['fold5']
dataset_teste = config['DATASET']['teste']
output_dir = config['OUTPUT_DIR']
num_folds = config['NUM_FOLDS']

best_F1_scores = {}
best_conf_scores = {}
result_data = [['fold','score','F1']]

for i in range(1,num_folds+1,1):
    if i == 1:
        dataset = dataset_fold1
    if i == 2:
        dataset = dataset_fold2
    if i == 3:
        dataset = dataset_fold3
    if i == 4:
        dataset = dataset_fold4
    if i == 5:
        dataset = dataset_fold5

    #train model
    model = YOLO(pre_trained_model)
    result_path_train = experiment_name+'/fold'+str(i)+'/train'
    results_train = TrainModel(model=model,
                            dataset=dataset_teste, #change for dataset when finish testing code
                            experiment_name=result_path_train,
                            epochs=epochs,
                            patience=patience,
                            img_size=img_size,
                            output_dir=output_dir)

    #search for best confidence score
    model_to_evaluate = YOLO(str(results_train.save_dir) + '/weights/best.pt')
    result_path_val = experiment_name+'/fold'+str(i)+'/val/iter'
    best_conf_score, best_F1_score = ConfidenceThresholdOptimization(model=model_to_evaluate,
                                                    dataset=dataset_teste, #change for dataset when finishin testing code
                                                    output_dir=output_dir,
                                                    experiment_name=result_path_val)

    #Saving result metrics
    result_data.append([i,best_conf_score,best_F1_score])
    #Writing file
    metrics_file_path = output_dir+experiment_name+'/fold'+str(i)+'/metrics.txt'
    with open(metrics_file_path, mode='w') as file:
        file.write(f"Fold {i}\nscore: {best_conf_score}\nF1: {best_F1_score}")

#Saving metrics of all folds
csv_file_path = output_dir+'/'+experiment_name+'/metrics.csv'
# Open the file in write mode
with open(csv_file_path, mode='w', newline='') as file:
    # Create a csv.writer object
    writer = csv.writer(file)
    # Write data to the CSV file
    writer.writerows(result_data)

print("Training workflow completed.")
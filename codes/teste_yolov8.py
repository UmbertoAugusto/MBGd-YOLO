from ultralytics import YOLO
import yaml

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

#training
model = YOLO(pre_trained_model)
model.info()
results = model.train(name=experiment_name,
                      data=dataset_teste,
                      epochs=epochs,
                      imgsz=img_size,
                      patience=patience,
                      project=output_dir)

print(type(results.save_dir),str(results.save_dir))
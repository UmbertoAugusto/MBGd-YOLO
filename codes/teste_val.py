from ultralytics import YOLO

model = YOLO("yolov8n.pt")
#/home/umberto.pereira/Mosquitoes/ultralytics/ultralytics/cfg/datasets/coco8.yaml
metrics = model.val(data='/home/umberto.pereira/Mosquitoes/YOLO/configs/datasets_configs/dataset_test.yaml',
                    project = '/home/umberto.pereira/Mosquitoes/YOLO/outputs',
                    name = 'val_testando',
                    conf = 0.5)
print(metrics.box.f1)
recall = metrics.box.r
precision = metrics.box.p
print(recall,precision)
print(metrics.box.nc)
my_f1 = []
for i in range(len(recall)):
    if precision[i]==0 and recall[i]==0:
        my_f1.append(0)
    else:
        f1_i = (2*precision[i]*recall[i])/(precision[i]+recall[i])
        my_f1.append(round(f1_i,5))
print(my_f1)
from ultralytics import YOLO

def TrainModel (model,dataset,experiment_name,epochs,patience,img_size,output_dir):
    '''Realiza o trainamento do modelo.'''
    results = model.train(name=experiment_name,
                      data=dataset,
                      epochs=epochs,
                      imgsz=img_size,
                      patience=patience,
                      project=output_dir)
    return results

def ConfidenceThresholdOptimization (model,dataset,output_dir,experiment_name):
    '''Realiza a busca pelo melhor valor para o Confidence Threshold.'''
    grid = range(10,102,2)
    best_F1 = 0
    best_tau = 0
    for x in grid:
        tau = x/100 #correcao para valores de 0.1 ate 1.0
        metrics = model.val(data=dataset,
                            project = output_dir,
                            name = experiment_name,
                            conf = tau)
        if len(metrics.box.f1)>0:
            F1 = metrics.box.f1[0]
        else:
            F1 = 0
        if F1 > best_F1:
            best_tau = tau
    return best_tau, best_F1
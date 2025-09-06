from ultralytics import YOLO
import pandas as pd

def TrainModel (model,dataset,experiment_name,hyp_params,epochs,patience,output_dir):
    '''Realiza o trainamento do modelo.'''
    results = model.train(name=experiment_name,
                        device=0, #mudar isso para vir do config file depois de testar
                        data=dataset,
                        cfg=hyp_params,
                        epochs=epochs,
                        patience=patience,
                        project=output_dir,
                        optimizer="Adam")
    return results

def ConfidenceThresholdOptimization (model,dataset,output_dir,experiment_name,verbose=True):
    '''Realiza a busca pelo melhor valor para o Confidence Threshold.'''
    grid = range(10,102,2)
    best_F1 = 0
    best_tau = 0
    for x in grid:
        tau = x/100 #correcao para valores de 0.1 ate 1.0
        metrics = model.val(data=dataset,
                            split='val',
                            project = output_dir,
                            name = experiment_name,
                            conf = tau,
                            imgsz = 640,
                            device = 0,
                            batch=32,
                            save_json=True)
        if len(metrics.box.f1)>0:
            F1 = metrics.box.f1[0]
        else:
            F1 = 0
        if verbose == True:
            print("////////////////////////////////////////////////")
            print("F1 PARA TAU = ", tau, ": ", F1)
            print("////////////////////////////////////////////////")
        if F1 >= best_F1:
            best_tau = tau
            best_F1 = F1
    return best_tau, best_F1

def EvaluatingConfidenceScores(model,dataset,output_dir,experiment_name,verbose=True):
    '''Realiza validacao para diferentes valores do Confidence Threshold.'''
    grid = range(10,102,2)
    results_data = [['score','F1']]
    for x in grid:
        tau = x/100 #correcao para valores de 0.1 ate 1.0
        metrics = model.val(data=dataset,
                            split='val',
                            project = output_dir,
                            name = experiment_name,
                            conf = tau,
                            imgsz = 640,
                            device = 0,
                            batch=32,
                            save_json=True)
        if len(metrics.box.f1)>0:
            F1 = metrics.box.f1[0]
        else:
            F1 = 0
        if verbose==True:
            print("////////////////////////////////////////////////")
            print("F1 PARA TAU = ", tau, ": ", F1)
            print("////////////////////////////////////////////////")
        results_data.append([tau,F1])
    return results_data

def GetBestConfScore(outer_loop_test_fold,output_dir,experiment_name):
    '''Le arquivos csv com confidence score e F1 para achar o confidence score que maximiza a media de F1.'''
    #getting paths
    data = []
    for i in range(1,6):
        if i == int(outer_loop_test_fold):
            continue
        results_path = f'{output_dir}/{experiment_name}/outer_test_fold{outer_loop_test_fold}/inner_val_fold{i}/val_metrics.csv'
        data.append(pd.read_csv(results_path))
    #calculating score that maximizes mean F1
    all_data = pd.concat(data,ignore_index=True)
    mean_f1_by_score = all_data.groupby('score')['F1'].mean()
    best_conf_score = mean_f1_by_score.idxmax()
    return best_conf_score

def TestModel (model,dataset,output_dir,experiment_name,conf_score,verbose=True):
    metrics = model.val(data=dataset,
                        split='test',
                        project = output_dir,
                        name = experiment_name,
                        conf = conf_score,
                        imgsz = 640,
                        device = 0,
                        batch=32,
                        save_json=True)
    
    confusion_matrix = metrics.confusion_matrix.matrix
    tp = confusion_matrix[0,0]
    fp = confusion_matrix[0,1]
    fn = confusion_matrix[1,0]

    if tp + fp == 0:
        pr = 0
    else:
        pr = tp / (tp + fp)
    if tp + fn == 0:
        rc = 0
    else:
        rc = tp / (tp + fn)
    if pr + rc == 0:
        F1 = 0
    else:
        F1 = (2 * pr * rc) / (pr + rc)

    mAP50_95 = metrics.box.map
    mAP50 = metrics.box.map50

    if verbose == True:
        print("////////////////////////////////////////////////")
        print("TEST results using confidence score = ", conf_score)
        print('True Positives: ',tp)
        print('False Positives: ',fp)
        print('False Negatives: ',fn)
        print('Precision: ',pr)
        print('Recall: ',rc)
        print('F1: ',F1)
        print('mAP@50-95: ',mAP50_95)
        print('mAP@50: ',mAP50)
        print("////////////////////////////////////////////////")
    return tp,fp,fn,pr,rc,F1,mAP50_95,mAP50
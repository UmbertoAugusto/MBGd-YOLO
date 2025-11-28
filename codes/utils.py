from ultralytics import YOLO
import pandas as pd
from post_processing import PostProcessingTiledImages
import os
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

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

def ConfidenceThresholdOptimization(
    model,
    dataset,
    output_dir,
    experiment_name,
    database,
    annotations_json_path=None,
    verbose=True,
    device=0,
    batch=32,
    imgsz=640,
):
    """
    #1 Varredura 1: taus = 0.10, 0.20, ..., 0.90  (9 avaliações)
    #2 Varredura fina:   em torno do melhor tau da grossa, de (tau-0.08) a (tau+0.08) passo 0.02,
                         limitado a [0.10, 0.90], excluindo o centro (8 avaliações)
    Total de 17 avaliações.
    """

    def eval_tau_for_integer_images(tau):
        results = model.val(
            data=dataset,
            split='val',
            project=output_dir,
            name=f"{experiment_name}_tau_{str(tau).replace('.', 'p')}",
            conf=float(tau),
            imgsz=imgsz,
            device=device,
            batch=batch,
            save_json=True,
        )

        if database.lower() == "integer":
            F1 = results.box.f1[0] if hasattr(results, "box") and len(getattr(results.box, "f1", [])) > 0 else 0.0
            return float(F1), False

        else:
            raise ValueError("database deve ser 'integer'.")

    best_tau, best_F1 = 0.0, 0.0

    if database.lower() == "integer":

        best_iou = None # usado apenas para database = "tiled"

        # PRIMEIRO LOOP 
        coarse_taus = [round(x/100, 2) for x in range(10, 100, 10)]  # 0.10..0.90
        for tau in coarse_taus:
            F1, stop = eval_tau_for_integer_images(tau)
            if verbose:
                print(f"[COARSE] tau={tau:.2f}  F1={F1:.6f}")
            if F1 >= best_F1:
                best_F1, best_tau = F1, tau
            if stop:
                break 

        # SEGUNDO LOOP
        f_start = max(0.10, round(best_tau - 0.08, 2))
        f_end   = min(0.90, round(best_tau + 0.08, 2))
        fine_taus = []
        t = f_start
        while t <= f_end + 1e-9:
            t = round(t, 2)
            if abs(t - best_tau) > 1e-12:  # exclui o centro, pois ja avaliado
                fine_taus.append(t)
            t += 0.02

        for tau in fine_taus:
            F1, _ = eval_tau_for_integer_images(tau)  # na fase fina, não precisamos interromper ao faltar preds
            if verbose:
                print(f"[FINE]   tau={tau:.2f}  F1={F1:.6f}")
            if F1 >= best_F1:
                best_F1, best_tau = F1, tau

    elif database.lower() == "tiled":
        print("Rodando inferência de base (conf=0.01) para 'tiled'...")
        base_results = model.val(
                            data=dataset,
                            split='val',
                            project=output_dir,
                            name=f"{experiment_name}_base_inference", # Nome único
                            conf=0.01,  # <-- MUITO IMPORTANTE: conf baixo para salvar TUDO
                            imgsz=imgsz,
                            device=device,
                            batch=batch,
                            save_json=True,
                        )
        # Le arquivo com previsoes
        save_dir = getattr(base_results, "save_dir", None)
        json_preds_path = None if save_dir is None else os.path.join(save_dir, "predictions.json")
        if not json_preds_path or not os.path.exists(json_preds_path):
            print(f"!!! ERRO: predictions.json não foi criado em {save_dir}")
            raise FileNotFoundError(f"Arquivo predictions.json não encontrado em {save_dir}")
        
        print(f"Inferência base concluída. JSON salvo em: {json_preds_path}")

        # Define espaco de busca para otimizacao Bayesiana 
        space = [
            Real(0.1, 0.98, name='conf_thresh'),
            Real(0.05, 0.9, name='iou_thresh')
        ]

        @use_named_args(space)
        def funcao_objetivo(**params):
            conf_k = params['conf_thresh']
            iou_k = params['iou_thresh']

            metrics = PostProcessingTiledImages(
                        pred_json_path=json_preds_path,
                        original_annotations_json_path=annotations_json_path,
                        confidence_threshold=float(conf_k),
                        iou_thresh_tp_fp=float(iou_k)
                        )

            F1 = float(metrics.get("F1", 0.0))

            if verbose:
                print(f"[OB] Testando: conf={conf_k:.4f}, iou={iou_k:.4f} -> F1={F1:.6f}")

            # skopt minimiza, então retornamos o F1 negativo para MAXIMIZAR
            return -F1
        
        # Faz otimizacao Bayesiana
        resultado_ob = gp_minimize(
                        func=funcao_objetivo,
                        dimensions=space,
                        n_calls=100,  # Número de iterações
                        random_state=0
                        )
        
        # Pega os melhores resultados
        best_tau = resultado_ob.x[0]
        best_iou = resultado_ob.x[1]
        best_F1 = -resultado_ob.fun # skopt minimiza e antes tinhamos invertido, agora volta para sinal original
        if verbose:
            print("\n--- Otimização Bayesiana Concluída ---")
            print(f"Melhor F1: {best_F1:.6f}")
            print(f"Melhor conf_thresh (tau): {best_tau:.4f}")
            print(f"Melhor iou_thresh: {best_iou:.4f}")

    else:
        raise ValueError("database deve ser 'integer' ou 'tiled'.")

    return best_tau, best_F1, best_iou

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

def TestModel (model,dataset,output_dir,experiment_name,conf_score,decision_iou_threshold,database,annotations_json_path,verbose=True):
    results = model.val(data=dataset,
                        split='test',
                        project = output_dir,
                        name = experiment_name,
                        conf = conf_score,
                        imgsz = 640,
                        device = 0,
                        batch=32,
                        save_json=True)
    
    if database.lower() == "integer":
        confusion_matrix = results.confusion_matrix.matrix
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

    elif database.lower() == "tiled":
        save_directory = results.save_dir
        json_preds_path = f"{save_directory}/predictions.json"
        metrics = PostProcessingTiledImages(pred_json_path = json_preds_path,
                                            original_annotations_json_path = annotations_json_path,
                                            confidence_threshold = conf_score,
                                            iou_thresh_tp_fp = decision_iou_threshold)
        tp = metrics['TP']
        fp = metrics['FP']
        fn = metrics['FN']
        pr = metrics['Precision']
        rc = metrics['Recall']
        F1 = metrics['F1']

    if verbose == True:
        print("////////////////////////////////////////////////")
        print("TEST results using confidence score = ", conf_score)
        print('True Positives: ',tp)
        print('False Positives: ',fp)
        print('False Negatives: ',fn)
        print('Precision: ',pr)
        print('Recall: ',rc)
        print('F1: ',F1)
        print("////////////////////////////////////////////////")
    return tp,fp,fn,pr,rc,F1
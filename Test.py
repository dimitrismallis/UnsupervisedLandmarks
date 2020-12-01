from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import numpy as np
from Train_options import Options
from Databases import Database
from FanClass import FAN_Model
import Utils
from Utils import LogText
import imgaug.augmenters as iaa
import resource
import yaml
import evaluation

def test(experiment_options,experiment_name,dataset_name,number_of_workers,log_path):
    step=2
    hyperparameters=experiment_options.GetHyperparameters(2,dataset_name)

    # training parameters
    number_of_clusters= hyperparameters.number_of_clusters
    confidence_thres_FAN=hyperparameters.confidence_thres_FAN
    iter_before_merging = hyperparameters.iter_before_merging
    batchSize= hyperparameters.batchSize
    weight_decay= hyperparameters.weight_decay
    lr= hyperparameters.lr
    batch_multiplier= hyperparameters.batch_multiplier
    number_of_clusters= hyperparameters.number_of_clusters
    totalIterations= hyperparameters.totalIterations
    lrstep=hyperparameters.lrstep
    confidence_thres_FAN=hyperparameters.confidence_thres_FAN
    clusteroverlap=hyperparameters.clusteroverlap

    Utils.initialize_log_dirs(experiment_name,log_path)
    checkpoint=Utils.GetPathsEval(experiment_name,log_path)

    FAN = FAN_Model(number_of_clusters,None,experiment_name,confidence_thres_FAN,log_path,step)
    FAN.init_secondstep(lr,weight_decay,batch_multiplier,number_of_clusters,lrstep,clusteroverlap)

    FAN.load_trained_secondstep_model(checkpoint_filename=checkpoint)

    evaluation_database = Database(dataset_name, number_of_clusters, test=True, function_for_dataloading=Database.get_FAN_secondStep_evaluation)
    evaluation_dataloader = DataLoader(evaluation_database, batch_size=10, shuffle=False,num_workers=number_of_workers, drop_last=True)

    keypoints=FAN.Get_labels_for_evaluation(evaluation_dataloader)

    return keypoints


if __name__=="__main__":
    torch.manual_seed(1993)
    torch.cuda.manual_seed_all(1993)
    np.random.seed(1993)
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

    experiment_options=Options()
    global args
    args = experiment_options.args  

    #load paths
    with open('paths/main.yaml') as file:
        paths = yaml.load(file, Loader=yaml.FullLoader)
    log_path=paths['log_path']

    # config parameters
    args = experiment_options.args
    experiment_name=args.experiment_name
    dataset_name = args.dataset_name
    number_of_workers = args.num_workers

    keypoints=test(experiment_options,experiment_name,dataset_name,number_of_workers,log_path)

    evaluator=evaluation.Evaluator(dataset_name,experiment_name,log_path)
    evaluator.Evaluate(keypoints)


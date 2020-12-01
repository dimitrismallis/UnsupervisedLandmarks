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


def main():
    step=2
    experiment_options=Options()
    global args
    args = experiment_options.args

    # config parameters
    args = experiment_options.args
    experiment_name=args.experiment_name
    dataset_name = args.dataset_name
    number_of_workers = args.num_workers
    resume =args.resume

    hyperparameters=experiment_options.GetHyperparameters(step,dataset_name)

    # training parameters
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

    #load paths
    with open('paths/main.yaml') as file:
        paths = yaml.load(file, Loader=yaml.FullLoader)
    log_path=paths['log_path']
    
    Utils.initialize_log_dirs(experiment_name,log_path)

    LogText(f"Experiment Name {experiment_name}\n"
                f"Database {dataset_name}\n"
                "Training Parameters \n"
                f"Batch size {batch_multiplier*batchSize} \n"
                f"Learning  rate {lr} \n"
                f"Weight Decay {weight_decay} \n"
                f"Iterations Before Mergins {iter_before_merging} \n"
                f"Total Iterations {totalIterations} \n"
                f"Number of Clusters {number_of_clusters} \n"
                , experiment_name,log_path)

    LogText("Training of Second step begins", experiment_name,log_path)

    
    #augmentations for Second step
    augmentations = iaa.Sequential([
        iaa.Sometimes(0.5,
                      iaa.GaussianBlur(sigma=(0, 0.5))
                      ),
        iaa.ContrastNormalization((0.75, 1.5)),
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
        iaa.Multiply((0.8, 1.2), per_channel=0.2),

        iaa.Sometimes(0.5,
                      iaa.MultiplyHueAndSaturation((0.5, 1.5), per_channel=True),
                      ),
        iaa.Affine(
            scale={"x": (0.7, 1.3), "y": (0.7, 1.3)},
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            rotate=(-40, 40),
        )
    ])

    criterion = nn.MSELoss(reduce=False).cuda()

    #model initialisation from weights of the first step
    FAN = FAN_Model(number_of_clusters,criterion,experiment_name,confidence_thres_FAN,log_path,step)

    if(resume):
        FAN.init_secondstep(lr,weight_decay,batch_multiplier,number_of_clusters,lrstep,clusteroverlap)
        path_to_checkpoint,path_to_keypoints=Utils.GetPathsResumeSecondStep(experiment_name,log_path)
        FAN.load_trained_secondstep_model(path_to_checkpoint)
        keypoints = Utils.load_keypoints(path_to_keypoints)
    else:
        path_to_checkpoint,path_to_keypoints=Utils.GetPathsTrainSecondStep(experiment_name,log_path)
        FAN.init_secondstep(lr,weight_decay,batch_multiplier,number_of_clusters,lrstep,clusteroverlap,path_to_checkpoint=path_to_checkpoint)
        keypoints = Utils.load_keypoints(path_to_keypoints)
        FAN.RemoveSmallClusters(keypoints)

    #initial dataloader
    dataset = Database( dataset_name, FAN.number_of_clusters, image_keypoints=keypoints,
            function_for_dataloading=Database.get_FAN_secondStep_train, augmentations=augmentations)
    dataloader = DataLoader(dataset, batch_size=batchSize, shuffle=True, num_workers=number_of_workers,drop_last=True)

    epochsbetweenMerging=0
    while FAN.iterations < totalIterations:
        FAN.Train_step2(dataloader)

        #merging operation is performed initially after $iter_before_merging iterations and
        #after that every epoch
        if (FAN.iterations > iter_before_merging and epochsbetweenMerging==0):
            
            LogText("Performing Cluster Merging", experiment_name,log_path)

            #create dataloader for cluster merge operation
            dataset_merging = Database(dataset_name, FAN.number_of_clusters,image_keypoints=keypoints,function_for_dataloading=Database.get_FAN_inference)
            dataloader_merging = DataLoader(dataset_merging, batch_size=batchSize, shuffle=False, num_workers=number_of_workers, drop_last=False)

            #form new set of keypoints with merged clusters
            keypoints = FAN.MergeClusters(dataloader_merging, keypoints)
            
            epochsbetweenMerging=1
            #update training dataloader with new set of keypoints
            dataset = Database( dataset_name, FAN.number_of_clusters, image_keypoints=keypoints,
                               function_for_dataloading=Database.get_FAN_secondStep_train, augmentations=augmentations)
            dataloader = DataLoader(dataset, batch_size=batchSize, shuffle=True, num_workers=number_of_workers,
                                    drop_last=True)

        #merging is performed every 2 epochs (epochsbetweenMerging+1)
        elif(FAN.iterations > iter_before_merging and epochsbetweenMerging>0):
            epochsbetweenMerging=epochsbetweenMerging-1



if __name__=="__main__":
    torch.manual_seed(1993)
    torch.cuda.manual_seed_all(1993)
    np.random.seed(1993)
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
    main()

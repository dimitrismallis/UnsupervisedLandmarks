from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from Train_options import Options
from SuperPoint import SuperPoint
from Databases import Database
import Utils
from Utils import LogText
from FanClass import FAN_Model
import resource
import imgaug.augmenters as iaa
import torch.nn as nn
import yaml



def main():
    step=1
    experiment_options=Options()
    global args

    # config parameters
    args = experiment_options.args
    experiment_name=args.experiment_name
    dataset_name = args.dataset_name
    number_of_workers = args.num_workers
    resume =args.resume

    hyperparameters=experiment_options.GetHyperparameters(step,dataset_name)
    # training parameters
    batchSize = hyperparameters.batchSize
    weight_decay = hyperparameters.weight_decay
    lr = hyperparameters.lr
    number_of_clusters = hyperparameters.number_of_clusters
    number_of_clustering_rounds=hyperparameters.number_of_clustering_rounds
    nms_thres_superpoint=hyperparameters.nms_thres_superpoint
    confidence_thres_superpoint=hyperparameters.confidence_thres_superpoint
    use_box=hyperparameters.use_box
    remove_superpoint_outliers_percentage=hyperparameters.remove_superpoint_outliers_percentage
    training_iterations_before_first_clustering=hyperparameters.training_iterations_before_first_clustering
    confidence_thres_FAN=hyperparameters.confidence_thres_FAN
    UseScales=hyperparameters.UseScales
    RemoveBackgroundClusters=hyperparameters.RemoveBackgroundClusters

    #load paths
    with open('paths/main.yaml') as file:
        paths = yaml.load(file, Loader=yaml.FullLoader)
    log_path=paths['log_path']
    path_to_superpoint_checkpoint=paths['path_to_superpoint_checkpoint']
    
    LogText(f"Experiment Name {experiment_name}\n"
            f"Database {dataset_name}\n"
            "Training Parameters: \n"
            f"Batch size {batchSize} \n"
            f"Learning  rate {lr} \n"
            f"Weight Decay {weight_decay} \n"
            f"Training iterations before first clustering  {training_iterations_before_first_clustering} \n"
            f"Number of clustering rounds {number_of_clustering_rounds} \n"
            f"FAN detection threshold {confidence_thres_FAN} \n"
            f"Number of Clusters {number_of_clusters} \n"
            f"Outlier removal  {remove_superpoint_outliers_percentage} \n"
            , experiment_name, log_path)

    LogText("Training of First step begins", experiment_name,log_path)

    #This funcion will create the directories /Logs and a /CheckPoints at log_path
    Utils.initialize_log_dirs(experiment_name,log_path)

    #augmentations for first step
    augmentations = iaa.Sequential([
    iaa.Sometimes(0.3,
                  iaa.GaussianBlur(sigma=(0, 0.5))
                  ),
    iaa.ContrastNormalization((0.85, 1.3)),
    iaa.Sometimes(0.5,
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5)
        )
    ,
    iaa.Multiply((0.9, 1.1), per_channel=0.2),
    iaa.Sometimes(0.3,
                  iaa.MultiplyHueAndSaturation((0.5, 1.5), per_channel=True),
                  ),
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
        rotate=(-40, 40),
    )
])
    
    #selection of the dataloading function
    superpoint_dataloading_function=Database.get_image_superpoint
    if(UseScales):
        superpoint_dataloading_function=Database.get_image_superpoint_multiple_scales

    superpoint= SuperPoint(number_of_clusters,
                           confidence_thres_superpoint,
                           nms_thres_superpoint,
                           path_to_superpoint_checkpoint,
                           experiment_name,
                           log_path,
                           remove_superpoint_outliers_percentage,
                           use_box,
                           UseScales,                                                                                                       
                           RemoveBackgroundClusters,
                           )

    superpoint_dataset=Database( dataset_name, number_of_clusters,
                                function_for_dataloading=superpoint_dataloading_function, augmentations=augmentations,use_box=use_box)
    dataloader = DataLoader(superpoint_dataset, batch_size=batchSize, shuffle=False, num_workers=number_of_workers,
                            drop_last=True)

    criterion = nn.MSELoss().cuda()

    FAN = FAN_Model(number_of_clusters, criterion, experiment_name,confidence_thres_FAN, log_path,step)
    FAN.init_firststep(lr,weight_decay,number_of_clusters,training_iterations_before_first_clustering)

    if(resume):
        path_to_checkpoint,path_to_keypoints=Utils.GetPathsResumeFirstStep(experiment_name,log_path)
        if(path_to_checkpoint is not None):
            FAN.load_trained_fiststep_model(path_to_checkpoint)
        keypoints=Utils.load_keypoints(path_to_keypoints)
    else:
        #get initial pseudo-groundtruth by applying superpoint on the training data
        keypoints=superpoint.CreateInitialPseudoGroundtruth(dataloader)

    dataset = Database( dataset_name, FAN.number_of_clusters, image_keypoints=keypoints,function_for_dataloading=Database.get_FAN_firstStep_train, augmentations=augmentations)
    dataloader = DataLoader(dataset, batch_size=batchSize, shuffle=True, num_workers=number_of_workers,drop_last=True)

    database_clustering = Database(dataset_name, FAN.number_of_clusters,function_for_dataloading=Database.get_FAN_inference)
    dataloader_clustering = DataLoader(database_clustering, batch_size=batchSize, shuffle=False,num_workers=number_of_workers, drop_last=True)

    for i in range(number_of_clustering_rounds):
        
        FAN.Train_step1(dataloader)

        keypoints=FAN.Update_pseudoLabels(dataloader_clustering,keypoints)

        dataset = Database(dataset_name, FAN.number_of_clusters, image_keypoints=keypoints,
                           function_for_dataloading=Database.get_FAN_firstStep_train, augmentations=augmentations)
        dataloader = DataLoader(dataset, batch_size=batchSize, shuffle=True, num_workers=number_of_workers,
                                drop_last=True)




if __name__=="__main__":
    torch.manual_seed(1993)
    torch.cuda.manual_seed_all(1993)
    np.random.seed(1993)
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
    main()

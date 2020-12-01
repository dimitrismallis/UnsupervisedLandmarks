import argparse
import yaml
from Train_options import Options
import Utils
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2   
from Databases import Database
import Test
import numpy as np
import random
from Utils import LogText

def main():
    parser = argparse.ArgumentParser(description='Unsupervised Learning of Object Landmarks via Self-Training Correspondence (NeurIPS20)')
    parser.add_argument('--visualisation',choices=['Step1_Clusters','Step2','Step1_Keypoints'])
    parser.add_argument('--dataset_name', choices=['CelebA','LS3D','Human3.6'], help='Select training dataset')
    parser.add_argument('--num_workers', default=16, help='Number of workers',type=int)
    parser.add_argument('--experiment_name', help='Name of experiment you from which checkpoint or groundtruth is going to be loaded')

    args=parser.parse_args()

    with open('paths/main.yaml') as file:
        paths = yaml.load(file, Loader=yaml.FullLoader)
    log_path=paths['log_path']

    experiment_options=Options(useparser=False)
    hyperparameters=experiment_options.GetHyperparameters(1,args.dataset_name)

    Utils.initialize_log_dirs(args.experiment_name,log_path)
    if(args.visualisation == 'Step1_Clusters'):
        path_to_keypoints=Utils.GetPathsForClusterVisualisation(args.experiment_name,log_path)
        keypoints=Utils.load_keypoints(path_to_keypoints)
        ShowClusters( keypoints, log_path, args.experiment_name,hyperparameters.number_of_clusters,args.dataset_name)

    if(args.visualisation == 'Step1_Keypoints'):
        path_to_keypoints=Utils.GetPathsForClusterVisualisation(args.experiment_name,log_path)
        keypoints=Utils.load_keypoints(path_to_keypoints)
        ShowKeypoints( keypoints, log_path, args.experiment_name,hyperparameters.number_of_clusters,args.dataset_name)


    if(args.visualisation == 'Step2'):
        keypoints=Test.test(experiment_options,args.experiment_name,args.dataset_name,args.num_workers,log_path)
        ShowVisualRes(keypoints,log_path, args.experiment_name,hyperparameters.number_of_clusters,args.dataset_name)


def ShowVisualRes(keypoints,log_path,experiment_name,number_of_clusters,dataset_name):

    fig = plt.figure(figsize=(34,55))
    gs1 = gridspec.GridSpec(13, 8)
    gs1.update(wspace=0.0, hspace=0.0)
    filenames=[k for k in keypoints.keys() if keypoints[k]['is_it_test_sample']]
    filenames.sort()
    filenames=filenames[:13*8]
    dataset = Database( dataset_name, number_of_clusters,test=True)
    for i in range(len(filenames)):

        ax = plt.subplot(gs1[i])
        plt.axis('off')
        pointstoshow = keypoints[filenames[i]]['prediction']
        image = dataset.getimage_FAN(dataset, filenames[i])
        ax.imshow(image)
        colors = [Utils.colorlist[int(i)] for i in np.arange(len(pointstoshow))]
        ax.scatter(pointstoshow[:, 0], pointstoshow[:, 1], s=400, c=colors, marker='P',edgecolors='black', linewidths=0.3)
    fig.show()
    fig.savefig(log_path+ 'Logs/' + experiment_name + f'/Step2.jpg')
    LogText(f"Step2 results created in {log_path+ 'Logs/' + experiment_name + f'/Step2.jpg'}", experiment_name,log_path)


def ShowClusters(keypoints,log_path,experiment_name,number_of_clusters,dataset_name):
    dataset = Database( dataset_name, number_of_clusters )

    image_names=list(keypoints.keys())
    random.shuffle(image_names)

    for cluster_number in range(number_of_clusters):
        
        counter_figureimages=0
        counter_datasetimages=0

        fig, subplots= plt.subplots(8,8,figsize=(15,15))
        subplots=subplots.reshape(-1)
        fig.subplots_adjust(wspace=0,hspace=0)

        for s in subplots:
            s.set_axis_off()

        while counter_figureimages<64:

            #for the case where cluster has less than 64 instances
            if(counter_datasetimages>len(keypoints)-1):
                fig.savefig(log_path+ 'Logs/' + experiment_name + f'/Cluster{cluster_number}.jpg')
                break
                
            imagename=image_names[counter_datasetimages]
            imagepoints = keypoints[imagename]

            #if cluster exists in image
            if(sum(imagepoints[:, 2]==cluster_number)>0):
                image = dataset.getimage_FAN(dataset,imagename)
                ax=subplots[counter_figureimages]
                ax.imshow(image)
                ax.scatter(4*imagepoints[imagepoints[:, 2]==cluster_number,0], 4*imagepoints[imagepoints[:, 2]==cluster_number, 1])
                counter_figureimages+=1

            counter_datasetimages+=1
        fig.savefig(log_path+ 'Logs/' + experiment_name + f'/Cluster{cluster_number}.jpg')
        LogText(f"Cluster images created in {log_path+ 'Logs/' + experiment_name + f'/Cluster{cluster_number}.jpg'}", experiment_name,log_path)


def ShowKeypoints(keypoints,log_path,experiment_name,number_of_clusters,dataset_name):

    dataset = Database( dataset_name, number_of_clusters )
    count=0
    image_names=list(keypoints.keys())
    random.shuffle(image_names)

    fig, subplots= plt.subplots(8,8,figsize=(15,15))
    subplots=subplots.reshape(-1)
    fig.subplots_adjust(wspace=0,hspace=0)
    for s in subplots:
        s.set_axis_off()
        
    while count<8*8:
        imagepoints = keypoints[image_names[count]]
        image = dataset.getimage_FAN(dataset,image_names[count])
        ax=subplots[count]
        ax.imshow(image)
        ax.scatter(4*imagepoints[:,0], 4*imagepoints[:, 1])
        count+=1
    
    fig.savefig(log_path+ 'Logs/' + experiment_name + f'/Keypoints.jpg')
    LogText(f"Keypoint images created in {log_path+ 'Logs/' + experiment_name + f'/Keypoints.jpg'}", experiment_name,log_path)

if __name__=="__main__":
    main()


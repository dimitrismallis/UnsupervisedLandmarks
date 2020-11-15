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

def main():
    parser = argparse.ArgumentParser(description='Unsupervised Learning of Object Landmarks via Self-Training Correspondence (NeurIPS20)')
    parser.add_argument('--visualisation',choices=['cluster_examples','visual_results'], default='visual_results')
    parser.add_argument('--dataset_name', choices=['CelebA'], help='Select training dataset')
    parser.add_argument('--num_workers', default=16, help='Number of workers',type=int)
    parser.add_argument('--experiment_name', default='test')

    args=parser.parse_args()

    with open('paths/main.yaml') as file:
        paths = yaml.load(file, Loader=yaml.FullLoader)
    log_path=paths['log_path']

    if(args.dataset_name == 'CelebA'):
        with open('paths/CelebA.yaml') as file:
                paths = yaml.load(file, Loader=yaml.FullLoader)
        datapath = paths['datapath']

    experiment_options=Options()
    hyperparameters=experiment_options.GetHyperparameters(1,args.dataset_name)

    if(args.visualisation == 'cluster_examples'):
        path_to_checkpoint,path_to_keypoints=Utils.GetPathsFromExperiment(args.experiment_name,log_path)
        keypoints=Utils.load_keypoints(path_to_keypoints)
        ShowClusters( keypoints, log_path,datapath, args.experiment_name,hyperparameters.number_of_clusters,args.dataset_name)

    if(args.visualisation == 'visual_results'):
        keypoints=Test.test(experiment_options,args.experiment_name,args.dataset_name,args.num_workers)
        ShowVisualRes(keypoints,log_path,datapath, args.experiment_name,hyperparameters.number_of_clusters,args.dataset_name)


def ShowVisualRes(keypoints,log_path,datapath,experiment_name,number_of_clusters,dataset_name):

    fig = plt.figure(figsize=(34,55))
    gs1 = gridspec.GridSpec(13, 8)
    gs1.update(wspace=0.0, hspace=0.0)
    filenames=[k for k in keypoints.keys() if keypoints[k]['is_it_test_sample']]
    filenames.sort()
    filenames=filenames[:13*8]
    dataset = Database( dataset_name, number_of_clusters )
    for i in range(len(filenames)):

        ax = plt.subplot(gs1[i])
        plt.axis('off')
        pointstoshow = keypoints[filenames[i]]['prediction']
        image = dataset.getimage_FAN(dataset, filenames[i])
        ax.imshow(image)
        colors = [Utils.colorlist[int(i)] for i in np.arange(len(pointstoshow))]
        ax.scatter(pointstoshow[:, 0], pointstoshow[:, 1], s=400, c=colors, marker='P',edgecolors='black', linewidths=0.3)
    fig.show()
    fig.savefig(log_path+ 'Logs/' + experiment_name + f'/VisualResults.jpg')


def ShowClusters(keypoints,log_path,datapath,experiment_name,number_of_clusters,dataset_name):

    for k in range(number_of_clusters):
        dataset = Database( dataset_name, number_of_clusters )
        cluster=k
        count=0
        count2=0
        lst=list(keypoints.keys())
        fig, subplots= plt.subplots(8,8,figsize=(15,15))
        subplots=subplots.reshape(-1)
        fig.subplots_adjust(wspace=0,hspace=0)
        for s in subplots:
            s.set_axis_off()
        while count<8*8:
            if(count2>len(keypoints)-1):
                fig.savefig(log_path+ 'Logs/' + experiment_name + f'/cluster{cluster}.jpg')
                break
            imagepoints = keypoints[lst[count2]]
            if(sum(imagepoints[:, 2]==cluster)>0):
     
                image = dataset.getimage_FAN(dataset,lst[count2])
                ax=subplots[count]
                ax.imshow(image)
                ax.scatter(4*imagepoints[imagepoints[:, 2]==cluster,0], 4*imagepoints[imagepoints[:, 2]==cluster, 1])
                count+=1
            count2+=1
        fig.savefig(log_path+ 'Logs/' + experiment_name + f'/cluster{cluster}.jpg')



if __name__=="__main__":
    main()


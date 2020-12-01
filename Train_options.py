import argparse
import os
import types

class Options():

    def __init__(self,useparser=True):
        if(useparser):
            self._parser = argparse.ArgumentParser(description='Unsupervised Learning of Object Landmarks via Self-Training Correspondence (NeurIPS2020)')
            self.initialize()
            self.parse_args()


    def initialize(self):
        self._parser.add_argument('--experiment_name', default='Run1',help='Please assign a unique name for each experiment. Use the same name for both training set 1 and 2.')
        self._parser.add_argument('--dataset_name', choices=['CelebA', 'LS3D','Human3.6'], default='CelebA',help='Select training dataset')
        self._parser.add_argument('--num_workers', default=16, help='Number of workers',type=int)
        self._parser.add_argument('--resume', default=True, help='If True step1 and 2 will resume form last saved checkpoint and pseudogroundtruth.')


    def GetHyperparameters(self,step,dataset_name):
        
        hyperparameters=types.SimpleNamespace()
        hyperparameters.confidence_thres_FAN=0.2
        hyperparameters.weight_decay=1e-5
        hyperparameters.lr=5e-4

        if(step==1):
            hyperparameters.batchSize=16
            hyperparameters.confidence_thres_superpoint=0.1
            hyperparameters.nms_thres_superpoint=12
            hyperparameters.training_iterations_before_first_clustering=20000
            hyperparameters.number_of_clustering_rounds=30
            hyperparameters.UseScales=True
            hyperparameters.use_box=True

            if(dataset_name in ['Human3.6']):
                hyperparameters.number_of_clusters=250
                hyperparameters.remove_superpoint_outliers_percentage=0.4
                hyperparameters.RemoveBackgroundClusters=True

            elif(dataset_name in ['CelebA', 'LS3D']):
                hyperparameters.number_of_clusters=100
                hyperparameters.remove_superpoint_outliers_percentage=0.4
                hyperparameters.RemoveBackgroundClusters=False
                

        elif(step==2):
            hyperparameters.training_epochs=200
            hyperparameters.iter_before_merging=25000
            hyperparameters.batchSize=32
            hyperparameters.batch_multiplier=2
            hyperparameters.lrstep=[50000,65000]
            hyperparameters.totalIterations=80000
            hyperparameters.clusteroverlap=0.7
            if(dataset_name in ['Human3.6']):
                hyperparameters.number_of_clusters=250
            elif(dataset_name in ['CelebA', 'LS3D']):
                hyperparameters.number_of_clusters=100

        return hyperparameters


    def parse_args(self):
        self.args = self._parser.parse_args()

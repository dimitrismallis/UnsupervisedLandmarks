from Utils import  *
from FanModel import FAN

from FanModel import custom_weight_norm
import torch.nn as nn
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import numpy as np
import Utils
from Utils import LogText
import clustering
import faiss

from scipy.optimize import linear_sum_assignment

class FAN_Model():
    def __init__(self,numberofchannels,criterion,experiment_name,confidence_thres_FAN,log_path,step):
        self.model = Cuda(FAN(numberofchannels,step))
        self.criterion=criterion
        self.log_path=log_path
        self.experiment_name=experiment_name
        self.log_path=log_path
        self.confidence_thres_FAN=confidence_thres_FAN

    def init_firststep(self,lr,weight_decay,number_of_clusters,training_iterations_before_first_clustering):

        LogText(f"Training model initiated", self.experiment_name, self.log_path)
        self.weight_decay = weight_decay
        self.lr = lr
        self.training_iterations_before_first_clustering=training_iterations_before_first_clustering
        self.number_of_clusters=number_of_clusters
        
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.lr,  weight_decay=self.weight_decay)
        self.schedualer = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=5, T_mult=1,eta_min=5e-6)

        self.centroid = None
        self.margin = 0.8
        self.eps = 1e-9
        self.KmeansClustering = clustering.Kmeans(self.number_of_clusters)
        self.iterations=0


    def init_secondstep(self,lr,weight_decay,batch_multiplier,number_of_clusters,lrstep,clusteroverlap,path_to_checkpoint=None):
        self.iterations = 0
        self.epoch=0
        self.batch_multiplier=batch_multiplier
        self.weight_decay=weight_decay
        self.lr = lr
        self.lrstep=lrstep
        if(path_to_checkpoint is not None):
            LogText(f"Fan Initiated from weights of : {path_to_checkpoint}",self.experiment_name,self.log_path)
            checkpoint = torch.load(path_to_checkpoint, map_location='cpu')
            self.model.load_state_dict(checkpoint['state_dict'])
        self.number_of_clusters=number_of_clusters
        self.clusteroverlap=clusteroverlap
        self.active_channels = np.arange(self.number_of_clusters)
        newlayer1 = Cuda(nn.Conv2d(256, self.number_of_clusters, kernel_size=1, stride=1, padding=0))
        self.model._modules['l1'] = newlayer1

        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.lr,  weight_decay=self.weight_decay)
        self.schedualer = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.1)


    def load_trained_secondstep_model(self,checkpoint_filename):
        LogText(f"Pretrained Second Step model loaded from  : {checkpoint_filename}", self.experiment_name,self.log_path)
        checkpoint = torch.load(checkpoint_filename, map_location='cpu')
        self.iterations = checkpoint['iteration']

        self.active_channels = checkpoint['active_channels']

        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.schedualer.load_state_dict(checkpoint['schedualer'])



    def load_trained_fiststep_model(self,checkpoint_filename):
        LogText(f"Pretrained First Step model loaded from  : {checkpoint_filename}", self.experiment_name,self.log_path)
        checkpoint = torch.load(checkpoint_filename, map_location='cpu')
        self.iterations = checkpoint['iteration']
        self.centroid = checkpoint['centroid']

        if (self.centroid is not None):
            self.KmeansClustering=clustering.Kmeans(self.number_of_clusters,self.centroid)

        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.schedualer = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=5, T_mult=1,eta_min=5e-6)



    def Train_step1(self, dataloader):

        LogText(f"Training Begins", self.experiment_name, self.log_path)
        self.model.train()
        while(True):
            for i_batch, sample in enumerate(dataloader):

                self.optimizer.zero_grad()
                if (self.iterations % 2000 == 0):
                    LogText(f"Iterations : {self.iterations}", self.experiment_name, self.log_path)
                if( self.iterations == self.training_iterations_before_first_clustering):
                    LogText(f"Intial training stage completed", self.experiment_name, self.log_path)
                    self.iterations+=1
                    self.save_step1()
                    return

                if (self.iterations % 2000 == 0 and self.iterations > self.training_iterations_before_first_clustering):
                    self.schedualer.step()
                    LogText(f'Current Learning rate :'  +str(self.optimizer.param_groups[0]['lr']),self.experiment_name, self.log_path)
                    if (self.optimizer.param_groups[0]['lr'] == self.lr):
                        self.iterations+=1
                        self.save_step1()    
                        return

                input = Cuda(sample['image'])
                descriptorpairs = Cuda(sample['keypoints'])
                keypointHeatmaps = (Cuda(sample['keypointHeatmaps']))

                bsize=input.size(0)
                number_of_pairs=descriptorpairs.size(1)


                batchid = Cuda(
                    torch.arange(bsize)
                        .repeat(number_of_pairs)
                        .reshape(number_of_pairs,bsize)
                        .transpose(1, 0))


                target = Cuda(descriptorpairs[:, :, 4].reshape(-1).clone())

                output1_detector, output1_descriptor = self.model(input[:, 0:3, :, :])
                output2_detector, output2_descriptor = self.model(input[:, 3:, :, :])

                loss_detector1 = self.criterion(output1_detector, keypointHeatmaps[:, 0:1, :, :])
                loss_detector2 = self.criterion(output2_detector, keypointHeatmaps[:, 1:2, :, :])

                output1features = output1_descriptor[
                                  batchid.reshape(-1).long(),
                                  :,
                                  descriptorpairs[:, :, 1].reshape(-1).long(),
                                  descriptorpairs[:, :, 0].reshape(-1).long()]

                output2features = output2_descriptor[
                                  batchid.reshape(-1).long(),
                                  :,
                                  descriptorpairs[:, :, 3].reshape(-1).long(),
                                  descriptorpairs[:, :, 2].reshape(-1).long()]

                distances = (output2features[descriptorpairs[:, :, 0].reshape(-1) != -1]
                             - output1features[descriptorpairs[:, :, 0].reshape(-1) != -1]
                             ).pow(2).sum(1)

                descriptor_losses = (target[descriptorpairs[:, :, 0].reshape(-1) != -1].float() * distances
                                     +
                                     (1 + -1 * target[descriptorpairs[:, :, 0].reshape(-1) != -1]).float()
                                     * torch.nn.functional.relu(
                                                self.margin - (distances + self.eps).sqrt()).pow(2))

                descriptor_losses = descriptor_losses.mean()

                loss = 10 * descriptor_losses + loss_detector1 + loss_detector2
                loss.backward()
                self.optimizer.step()
                self.iterations+=1



    def Train_step2(self,dataloader):

        self.model.train()
        count = 0
        LogText(f"Epoch {self.epoch} Training Begins", self.experiment_name,self.log_path)
        for i_batch, sample in enumerate(dataloader):

            if (self.iterations>0  and self.iterations in self.lrstep):
                self.schedualer.step()
                LogText('LR ' + str(self.optimizer.param_groups[0]['lr']),self.experiment_name,self.log_path)
                self.iterations+=1

            if (count == 0):
                self.optimizer.zero_grad()
                count = self.batch_multiplier

            input = Cuda(sample['image'])
            GaussianShape = Cuda(sample['GaussianShape'])

            GaussianShape = GaussianShape[:, self.active_channels, :, :]

            heatmaps_with_keypoints = Cuda(sample['heatmaps_with_keypoints'])
            heatmaps_with_keypoints=heatmaps_with_keypoints[:, self.active_channels]

            output_shape = self.model(input)

            output_shape = output_shape[:, self.active_channels, :, :]

            loss = torch.mean(self.criterion(output_shape, GaussianShape)[heatmaps_with_keypoints])
            
            loss.backward()
            count -= 1
            if (count == 0):
                self.optimizer.step()
                self.iterations += 1

        LogText('Epoch '+ str(self.epoch) +' completed, iterations '+str(self.iterations),self.experiment_name,self.log_path)

        self.save_step2()
        self.epoch += 1



    def Update_pseudoLabels(self,dataloader,oldkeypoints=None):

        LogText(f"Clustering stage for iteration {self.iterations}", self.experiment_name, self.log_path)
        self.model.eval()

        imagesize=256
        heatmapsize=64
        numberoffeatures = 256
        buffersize = 500000
        # allocation of 2 buffers for temporal storing of keypoints and descriptors.
        Keypoint_buffer = torch.zeros(buffersize,3)
        Descriptor__buffer = torch.zeros(buffersize, numberoffeatures)

        # arrays on which we save buffer content periodically. Corresponding files are temporal and
        # will be deleted after the completion of the process
        CreateFileArray(self.log_path + 'CheckPoints/' + self.experiment_name + '/keypoints', 3)
        CreateFileArray(self.log_path + 'CheckPoints/' + self.experiment_name + '/descriptors', numberoffeatures)

        # intermediate variables
        first_index = 0
        last_index = 0
        buffer_first_index = 0
        buffer_last_index = 0
        keypoint_indexes = {}

        pointsperimage=0
        LogText(f"Inference of keypoints and descriptors begins", self.experiment_name, self.log_path)
        for i_batch, sample in enumerate(dataloader):
            input = Cuda(sample['image'])
            names = sample['filename']

            with torch.no_grad():
                output = self.model.forward(input)
            outputHeatmap = output[0]
            descriptors_volume = output[1]

            batch_keypoints = GetBatchMultipleHeatmap(outputHeatmap, self.confidence_thres_FAN)

            for i in range(input.size(0)):

                indexes = batch_keypoints[:, 0] == i
                sample_keypoints = batch_keypoints[indexes, 1:][:,:3]


                pointsperimage+=len(sample_keypoints)
                if(oldkeypoints is not None):
                    if(names[i] in oldkeypoints):
                        keypoints_previous_round=Cuda(torch.from_numpy(oldkeypoints[names[i]].copy())).float()
                        sample_keypoints=MergePoints(sample_keypoints,keypoints_previous_round)



                descriptors = GetDescriptors(descriptors_volume[i], sample_keypoints[:, :2],
                                             heatmapsize,
                                             heatmapsize)
          
                numofpoints = sample_keypoints.shape[0]
                last_index += numofpoints
                buffer_last_index += numofpoints

                Keypoint_buffer[buffer_first_index: buffer_last_index, :2] = sample_keypoints.cpu()[:,:2]
                Descriptor__buffer[buffer_first_index: buffer_last_index, :] = descriptors

                keypoint_indexes[names[i]] = [first_index, last_index]
                first_index += numofpoints
                buffer_first_index += numofpoints

              
            # periodically we store the buffer in file
            if buffer_last_index > int(buffersize * 0.8):
                AppendFileArray(np.array(Keypoint_buffer[:buffer_last_index]),
                                self.log_path + 'CheckPoints/' + self.experiment_name + '/keypoints')
                AppendFileArray(np.array(Descriptor__buffer[:buffer_last_index]),
                                self.log_path + 'CheckPoints/' + self.experiment_name + '/descriptors')

                Keypoint_buffer = torch.zeros(buffersize, 3)
                Descriptor__buffer = torch.zeros(buffersize, numberoffeatures)
                buffer_first_index = 0
                buffer_last_index = 0

        # store any keypoints left on the buffers
        AppendFileArray(np.array(Keypoint_buffer[:buffer_last_index]),self.log_path + 'CheckPoints/' + self.experiment_name + '/keypoints')
        AppendFileArray(np.array(Descriptor__buffer[:buffer_last_index]),self.log_path + 'CheckPoints/' + self.experiment_name + '/descriptors')

        # load handlers to the Keypoints and Descriptor files
        Descriptors, fileHandler1 = OpenreadFileArray(self.log_path + 'CheckPoints/' + self.experiment_name + '/descriptors')
        Keypoints, fileHandler2 = OpenreadFileArray(self.log_path + 'CheckPoints/' + self.experiment_name + '/keypoints')
        Keypoints = Keypoints[:, :]
        LogText(f"Keypoints Detected per image Only detector {pointsperimage / len(keypoint_indexes)}", self.experiment_name,self.log_path)
        LogText(f"Inference of keypoints and descriptors completed", self.experiment_name, self.log_path)
        LogText(f"Keypoints Detected per image {len(Keypoints)/len(keypoint_indexes)}", self.experiment_name, self.log_path)

        # we use a subset of all the descriptors for clustering based on the recomendation of the Faiss repository
        numberOfPointsForClustering = 500000

        descriptors = clustering.preprocess_features(Descriptors[:numberOfPointsForClustering])
        _,self.centroid=self.KmeansClustering.cluster(descriptors, verbose=False)
        

        self.KmeansClustering.clus.nredo = 1

        thresholds = self.GetThresholdsPerCluster(Descriptors)

        Image_Keypoints = {}

        averagepointsperimage = 0

        for image in keypoint_indexes:
            start, end = keypoint_indexes[image]
            keypoints = Keypoints[start:end, :]

            image_descriptors = clustering.preprocess_features(Descriptors[start:end])

            # calculate distance of each keypoints to each centroid
            distanceMatrix, clustering_assignments = self.KmeansClustering.index.search(image_descriptors,
                                                                                   self.number_of_clusters)

            distanceMatrix = np.take_along_axis(distanceMatrix, np.argsort(clustering_assignments), axis=-1)

            # assign keypoints to centroids using the Hungarian algorithm. This ensures that each
            # image has at most one instance of each cluster
            keypointIndex, clusterAssignment = linear_sum_assignment(distanceMatrix)

            tempKeypoints=np.zeros((len(keypointIndex),3))
            tempKeypoints = keypoints[keypointIndex]

            clusterAssignmentDistance = distanceMatrix[keypointIndex, clusterAssignment]

            clusterstokeep = np.zeros(len(clusterAssignmentDistance))
            clusterstokeep = clusterstokeep == 1

            # keep only points that lie in their below a cluster specific theshold
            clusterstokeep[clusterAssignmentDistance < thresholds[clusterAssignment]] = True

            tempKeypoints[:,2] =clusterAssignment

            Image_Keypoints[image] = tempKeypoints[clusterstokeep]

            averagepointsperimage+=sum(clusterstokeep)

        #initialise centroids for next clustering round
        self.KmeansClustering=clustering.Kmeans(self.number_of_clusters,self.centroid)
        LogText(f"Keypoints Detected per image {averagepointsperimage/len(Image_Keypoints)}", self.experiment_name, self.log_path)

        self.save_keypoints(Image_Keypoints, f'UpdatedKeypoints{self.iterations}.pickle')
        ClosereadFileArray(fileHandler1, self.log_path + 'CheckPoints/' + self.experiment_name + '/keypoints')
        ClosereadFileArray(fileHandler2, self.log_path + 'CheckPoints/' + self.experiment_name + '/descriptors')
        LogText(f"Clustering stage completed", self.experiment_name, self.log_path)
        return Image_Keypoints



    def MergeClusters(self, dataloader,Image_Keypoints):

        LogText('Predictions for evaluation FAN',self.experiment_name,self.log_path)
        self.model.eval()
        Image_shapeKeypoints = {}
        for i_batch, sample in enumerate(dataloader):
            input = Cuda(sample['image'])
            name = sample['filename']

            with torch.no_grad():
                output = self.model.forward(input)

            output = output[:, torch.from_numpy(self.active_channels)]

            bsize = output.size(0)
            for i in range(bsize):
                Image_shapeKeypoints[name[i]] = Utils.GetPointsFromHeatmaps(output[i])
                
        # get points per cluster
        points_per_cluster = np.zeros(self.number_of_clusters)
        for index in Image_Keypoints:
            cluster_assignments = Image_Keypoints[index][:, 2]
            points_per_cluster[cluster_assignments.astype(int)] += 1
        points_per_channel=points_per_cluster[self.active_channels]

        totalDistanceMatrix = np.zeros((len(self.active_channels),len(self.active_channels)))
        # get spatial distance between clusters
        numberofconfidentpointspercluster=np.zeros(len(self.active_channels))
        for image in Image_shapeKeypoints.keys():
            points = Image_shapeKeypoints[image].detach().cpu().numpy()
            distancematrix = squareform(pdist(points[:, :2]))
            numberofconfidentpointspercluster[points[:, 2] > 0.2]+=1
            distancematrix[points[:, 2] < self.confidence_thres_FAN, :] = 300
            distancematrix[:, points[:, 2] < self.confidence_thres_FAN] = 300
            #5.7 corresponds to nms of size 1 on the 64x64 dimension
            totalDistanceMatrix = totalDistanceMatrix + (distancematrix < 5.7).astype(int)

        confident_points_per_channel = np.diag(totalDistanceMatrix).copy()
        np.fill_diagonal(totalDistanceMatrix, 0)

        indexes_sorted=np.argsort(points_per_channel)[::-1]

        points_of_smaller_cluster = np.zeros((len(self.active_channels), len(self.active_channels)))
        for x in range(len(self.active_channels)):
            for y in range(len(self.active_channels)):
                points_of_smaller_cluster[x, y] = min(numberofconfidentpointspercluster[x],
                                                      numberofconfidentpointspercluster[y])

        indexes_channels_to_extend=np.array([])
        indexes_channels_merged=[]

        while(len(indexes_sorted)>0):
            channel=indexes_sorted[0]
            is_remaining=True
            for i in range(len(indexes_channels_to_extend)):
                element=indexes_channels_to_extend[i]
                if(totalDistanceMatrix[int(element),int(channel)]>self.clusteroverlap * points_of_smaller_cluster[int(element),int(channel)]):
                    indexes_channels_merged[i]=np.append(indexes_channels_merged[i],int(channel)).astype(int)
                    is_remaining=False
                    indexes_sorted = np.delete(indexes_sorted, 0)
                    break
            if(is_remaining):
                indexes_channels_to_extend=np.append(indexes_channels_to_extend,int(channel))
                indexes_channels_merged.append(np.array([]))
                indexes_sorted=np.delete(indexes_sorted,0)


        extendclusters=self.active_channels[indexes_channels_to_extend.astype(int)]
        clusters_merged=[]
        for el in indexes_channels_merged:
            clusters_merged.append(self.active_channels[el.astype(int)])


        pairs_to_keep=np.array([len(f)>0 for f in clusters_merged])
        extendclusters=extendclusters[pairs_to_keep].astype(int)
        clusters_merged=np.array(clusters_merged)[pairs_to_keep]

        count=0
        if (len(extendclusters) > 0):

            LogText("Clusters merged:",self.experiment_name,self.log_path)
            for s in range(len(extendclusters)):
                LogText(f"{extendclusters[s]}  -> {clusters_merged[s]}",self.experiment_name,self.log_path)

            # substitute merged clusters
            for index in Image_Keypoints:
                keypoint=Image_Keypoints[index]

                for p in range(len(extendclusters)):

                    indeces_of_keypoints_to_merge=np.in1d(keypoint[:, 2], clusters_merged[p] )
                    if (sum(indeces_of_keypoints_to_merge) ==0): 
                        continue
                    elif(sum(indeces_of_keypoints_to_merge)>0):

                        indeces_of_keypoints_to_merge = np.in1d(keypoint[:, 2], np.append(clusters_merged[p],extendclusters[p]))
                        
                        clusterinimage=keypoint[:, 2][indeces_of_keypoints_to_merge].astype(int)

                        index_of_bigger_cluster=np.argmax(points_per_cluster[clusterinimage])
                        cluster_to_remove=np.delete(clusterinimage,index_of_bigger_cluster)

                        indexes_to_keep=np.in1d(keypoint[:, 2],cluster_to_remove)==False
                        keypoint=keypoint[indexes_to_keep]
                        indeces_of_keypoints_to_merge = np.in1d(keypoint[:, 2], clusters_merged[p])
                        keypoint[:, 2][indeces_of_keypoints_to_merge] = extendclusters[p]

                Image_Keypoints[index] = keypoint


        self.active_channels = self.active_channels[indexes_channels_to_extend.astype(int)]
        self.active_channels.sort()

        LogText(f"Remaining Clusters: {len(self.active_channels)}",self.experiment_name,self.log_path)

        self.save_keypoints(Image_Keypoints, f'MergedKeypoints{self.iterations}.pickle')

        return Image_Keypoints



    def Get_labels_for_evaluation(self,dataloader):
        LogText('Predictions for evaluation FAN',self.experiment_name,self.log_path)
        self.model.eval()

        keypoints={}
        for i_batch, sample in enumerate(dataloader):

            input = Cuda(sample['image'])
            bsize = input.size(0)
            name = sample['filename']
            groundtruth=sample['groundtruth']
            is_test_sample=sample['is_it_test_sample']
            with torch.no_grad():
                output = self.model.forward(input)

            output = output[:,torch.from_numpy(self.active_channels)]

            for i in range(bsize):
                sampleKeypoints=Utils.GetPointsFromHeatmaps(output[i])[:,:3].detach().cpu().numpy()
                sampleKeypoints[sampleKeypoints[:,2]<self.confidence_thres_FAN]=np.nan
                sampleKeypoints=sampleKeypoints[:,:2]
                samplegroundtruth=groundtruth[i].detach().cpu().numpy()

                keypoints[name[i]]={'prediction':sampleKeypoints,'groundtruth':samplegroundtruth,'is_it_test_sample':is_test_sample[i]}

        

        self.save_keypoints(keypoints, f'EvaluateStep2Keypoints{self.iterations}.pickle')
        return keypoints





    def GetThresholdsPerCluster(self,Descriptors):

        rg = np.linspace(0, len(Descriptors), math.ceil(len(Descriptors) / 10000) + 1, dtype=int)
        distance_to_centroid_per_cluster = list([[] for i in range(self.number_of_clusters)])

        for i in range(len(rg) - 1):
            descriptors = clustering.preprocess_features(Descriptors[rg[i]:rg[i + 1], :][rg[i]:rg[i + 1]])
            distancesFromCenter, clustering_assingments = self.KmeansClustering.index.search(descriptors, 1)
            for point in range(len(clustering_assingments)):
                distance_to_centroid_per_cluster[int(clustering_assingments[point])].append(
                    distancesFromCenter[point][0])

        thresholds = np.zeros(self.number_of_clusters)

        for i in range(self.number_of_clusters):
            if (len(distance_to_centroid_per_cluster[i]) == 0):
                thresholds[i] = 0
            else:
                thresholds[i]=np.average(np.array(distance_to_centroid_per_cluster[i]))+np.std(distance_to_centroid_per_cluster[i])


        return thresholds


    #very small clusters with less than 500 points are removed from training of the second step
    def RemoveSmallClusters(self,keypoints):
        population=np.zeros(self.number_of_clusters)
        for k in keypoints.keys():
            temp=keypoints[k][:,2].astype(int)
            population[temp]+=1
        self.active_channels=self.active_channels[population>500]


    def save_step1(self):
        checkPointDirectory = self.log_path+ 'CheckPoints/' + self.experiment_name + '/'
        checkPointFileName = checkPointDirectory + f'{self.experiment_name}FirstStepIteration{self.iterations}' + '.pth'
        save_parameters = {
        'state_dict': self.model.state_dict(),
        'optimizer':  self.optimizer.state_dict(),
        'iteration':  self.iterations,
        'centroid': self.centroid
         }
        torch.save(save_parameters, checkPointFileName)


    def save_step2(self):
        checkPointDirectory = self.log_path+ 'CheckPoints/' + self.experiment_name + '/'
        checkPointFileName = checkPointDirectory + f'{self.experiment_name}SecondStepEpoch{self.epoch}' + '.pth'
        save_parameters = {
        'state_dict': self.model.state_dict(),
        'optimizer':  self.optimizer.state_dict(),
        'iteration':  self.iterations,
        'active_channels':self.active_channels,
        'schedualer':self.schedualer.state_dict()
         }
        torch.save(save_parameters, checkPointFileName)



    def save_keypoints(self,Image_Keypoints,filename):
        checkPointdir = self.log_path+ 'CheckPoints/' + self.experiment_name + '/'
        checkPointFile=checkPointdir+filename
        with open(checkPointFile, 'wb') as handle:
            pickle.dump(Image_Keypoints, handle, protocol=pickle.HIGHEST_PROTOCOL)






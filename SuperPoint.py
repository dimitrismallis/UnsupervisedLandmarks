import torch
from Utils import *
import torchvision
import math
import numpy as np
import faiss
from Utils import LogText
import clustering
from scipy.optimize import linear_sum_assignment
import imgaug.augmenters as iaa
import imgaug.augmentables.kps 


class SuperPoint():
    def __init__(self, number_of_clusters, confidence_thres_superpoint,nms_thres_superpoint,path_to_pretrained_superpoint,experiment_name,log_path,remove_superpoint_outliers_percentage,use_box=False,UseScales=False,RemoveBackgroundClusters=False):
        self.path_to_pretrained_superpoint=path_to_pretrained_superpoint
        self.use_box=use_box
        self.confidence_thres_superpoint=confidence_thres_superpoint
        self.nms_thres_superpoint=nms_thres_superpoint
        self.log_path=log_path
        self.remove_superpoint_outliers_percentage=remove_superpoint_outliers_percentage
        self.experiment_name=experiment_name
        self.number_of_clusters=number_of_clusters
        self.model = Cuda(SuperPointNet())
        self.UseScales=UseScales
        self.RemoveBackgroundClusters=RemoveBackgroundClusters
        if(self.UseScales):
            self.SuperpointUndoScaleDistill1 = iaa.Affine(scale={"x": 1 / 1.3, "y": 1 / 1.3})
            self.SuperpointUndoScaleDistill2 = iaa.Affine(scale={"x": 1 / 1.6, "y": 1 / 1.6})

        checkpoint = torch.load(path_to_pretrained_superpoint, map_location='cpu')
        
        self.model.load_state_dict(checkpoint['state_dict'])
        LogText(f"Superpoint Network from checkpoint {path_to_pretrained_superpoint}", self.experiment_name, self.log_path)
        
        self.softmax = torch.nn.Softmax(dim=1)
        self.pixelSuffle = torch.nn.PixelShuffle(8)
        self.model.eval()


    def CreateInitialPseudoGroundtruth(self, dataloader):

        LogText(f"Extraction of initial Superpoint pseudo groundtruth", self.experiment_name,self.log_path)

        imagesize=256
        heatmapsize=64
        numberoffeatures=256
        buffersize=500000

        #allocation of 2 buffers for temporal storing of keypoints and descriptors.
        Keypoint_buffer = torch.zeros(buffersize, 3)
        Descriptor__buffer = torch.zeros(buffersize, numberoffeatures)

        #arrays on which we save buffer content periodically. Corresponding files are temporal and
        #will be deleted after the completion of the process
        CreateFileArray(self.log_path+'CheckPoints/' +self.experiment_name + '/keypoints',3)
        CreateFileArray(self.log_path+'CheckPoints/' +self.experiment_name + '/descriptors', numberoffeatures)

        #intermediate variables
        first_index = 0
        last_index = 0
        buffer_first_index = 0
        buffer_last_index = 0
        keypoint_indexes = {}

        LogText(f"Inference of Keypoints begins", self.experiment_name, self.log_path)
        for i_batch, sample in enumerate(dataloader):
            input = Cuda(sample['image_gray'])
            names = sample['filename']
            bsize=input.size(0)

            if(self.UseScales):
                input=input.view(-1,1,input.shape[2],input.shape[3])

            with torch.no_grad():
                detectorOutput,descriptorOutput=self.GetSuperpointOutput(input)

            if(self.UseScales):
                detectorOutput=detectorOutput.view(bsize,-1,detectorOutput.shape[2],detectorOutput.shape[3])
                input=input.view(bsize,-1,input.shape[2],input.shape[3])
                descriptorOutput=descriptorOutput.view(bsize,-1,descriptorOutput.size(1),descriptorOutput.size(2),descriptorOutput.size(3))[:,0]
            for i in range(0, bsize):
                
                keypoints = self.GetPoints(detectorOutput[i].unsqueeze(0), self.confidence_thres_superpoint, self.nms_thres_superpoint)
                

                # import matplotlib
                # import matplotlib.pyplot as plt
                # fig, ax = plt.subplots(1)
                # ax.set_axis_off()
                # ax.imshow(input[i,0:1].detach().cpu().numpy().transpose(1,2,0),cmap='gray', vmin=0.0, vmax=1.0)
                # ax.scatter(keypoints[:, 0].cpu().detach().numpy(), keypoints[:, 1].cpu().detach().numpy())
                # # ax.scatter(bounding_box[[0,2]].cpu().detach().numpy(), bounding_box[[1,3]].cpu().detach().numpy())
                # plt.show()
                # fig.savefig(f'/home/SERILOCAL/d.mallis/Projects/UnsupervisedLandmarks/foo.jpg')
                # # fig.savefig(f'/home/SERILOCAL/d.mallis/Logs/test2/epoch15_{i}.jpg')
                
                if (self.RemoveBackgroundClusters):
                    bounding_box=sample['bounding_box'][i]
                    pointsinbox = torch.ones(len(keypoints))
                    pointsinbox[(keypoints[:, 0] < int(bounding_box[0]))] = -1
                    pointsinbox[(keypoints[:, 1] < int(bounding_box[1]))] = -1
                    pointsinbox[(keypoints[:, 0] > int(bounding_box[2]))] = -1
                    pointsinbox[(keypoints[:, 1] > int(bounding_box[3]))] = -1


                elif (self.use_box):
                    bounding_box=sample['bounding_box'][i]
                    pointsinbox = torch.ones(len(keypoints))
                    pointsinbox[(keypoints[:, 0] < int(bounding_box[0]))] = -1
                    pointsinbox[(keypoints[:, 1] < int(bounding_box[1]))] = -1
                    pointsinbox[(keypoints[:, 0] > int(bounding_box[2]))] = -1
                    pointsinbox[(keypoints[:, 1] > int(bounding_box[3]))] = -1
                    keypoints=keypoints[pointsinbox==1]


                # import matplotlib
                # import matplotlib.pyplot as plt
                # fig, ax = plt.subplots(1)
                # ax.set_axis_off()
                # ax.imshow(input[i,0:1].detach().cpu().numpy().transpose(1,2,0),cmap='gray', vmin=0.0, vmax=1.0)
                # ax.scatter(keypoints[:, 0].cpu().detach().numpy(), keypoints[:, 1].cpu().detach().numpy())
                # ax.scatter(bounding_box[[0,2]].cpu().detach().numpy(), bounding_box[[1,3]].cpu().detach().numpy())
                # plt.show()
                # fig.savefig(f'/home/SERILOCAL/d.mallis/Projects/UnsupervisedLandmarks/foo2.jpg')
                # # fig.savefig(f'/home/SERILOCAL/d.mallis/Logs/test2/epoch15_{i}.jpg')
                
                # import time
                

                descriptors = GetDescriptors(descriptorOutput[i], keypoints, imagesize, imagesize)

                #scale image keypoints to FAN resolution
                keypoints=dataloader.dataset.scaleforFAN(dataloader.dataset,names[i],keypoints)
                keypoints = ((heatmapsize/imagesize)*keypoints).round()

  
                last_index += len(keypoints)
                buffer_last_index += len(keypoints)

                Keypoint_buffer[buffer_first_index:buffer_last_index, :2] = keypoints
                Descriptor__buffer[buffer_first_index:buffer_last_index] = descriptors

                if (self.RemoveBackgroundClusters):
                    Keypoint_buffer[buffer_first_index:buffer_last_index, 2] = pointsinbox

                keypoint_indexes[names[i]] = [first_index, last_index]
                first_index += len(keypoints)
                buffer_first_index += len(keypoints)

            #periodically we store the buffer in file
            if buffer_last_index>int(buffersize*0.8):
                AppendFileArray(np.array(Keypoint_buffer[:buffer_last_index]),self.log_path+'CheckPoints/' +self.experiment_name + '/keypoints')
                AppendFileArray(np.array(Descriptor__buffer[:buffer_last_index]), self.log_path+'CheckPoints/' +self.experiment_name + '/descriptors')

                Keypoint_buffer = torch.zeros(buffersize, 3)
                Descriptor__buffer = torch.zeros(buffersize, numberoffeatures)
                buffer_first_index = 0
                buffer_last_index = 0

        LogText(f"Inference of Keypoints completed", self.experiment_name, self.log_path)
        #store any keypoints left on the buffers
        AppendFileArray(np.array(Keypoint_buffer[:buffer_last_index]), self.log_path+'CheckPoints/' +self.experiment_name + '/keypoints')
        AppendFileArray(np.array(Descriptor__buffer[:buffer_last_index]), self.log_path+'CheckPoints/' +self.experiment_name + '/descriptors')

        #load handlers to the Keypoints and Descriptor files
        Descriptors,fileHandler1=OpenreadFileArray(self.log_path+'CheckPoints/' +self.experiment_name + '/descriptors')
        Keypoints, fileHandler2 = OpenreadFileArray( self.log_path+'CheckPoints/' +self.experiment_name + '/keypoints')
        Keypoints = Keypoints[:, :]
        LogText(f"Keypoints Detected per image {len(Keypoints)/len(keypoint_indexes)}", self.experiment_name, self.log_path)

        #perform outlier detection
        inliersindexes=np.ones(len(Keypoints))==1
        if(self.remove_superpoint_outliers_percentage>0):
            inliersindexes=self.Indexes_of_inliers(Keypoints,Descriptors,buffersize)

        #extend outliers with background points for constant background datasets
        if (self.RemoveBackgroundClusters):
            foregroundpointindex=self.Indexes_of_BackgroundPoints(Keypoints,Descriptors,keypoint_indexes)
            inliersindexes = np.logical_and(inliersindexes, foregroundpointindex)

        LogText(f"Keypoints Detected per image(filtering) {sum(inliersindexes) / len(keypoint_indexes)}", self.experiment_name,self.log_path)
        #we use a subset of all the descriptors for clustering based on the recomendation of the Faiss repository
        numberOfPointsForClustering=500000

        LogText(f"Clustering of keypoints", self.experiment_name, self.log_path)
        #clustering of superpoint features
        KmeansClustering = clustering.Kmeans(self.number_of_clusters, centroids=None)
        descriptors = clustering.preprocess_features( Descriptors[:numberOfPointsForClustering][inliersindexes[:numberOfPointsForClustering]])
        KmeansClustering.cluster(descriptors, verbose=False)


        thresholds=self.GetThresholdsPerCluster( inliersindexes,Descriptors,KmeansClustering)


        Image_Keypoints = {}
        averagepointsperimage=0
        for image in keypoint_indexes:
            start,end=keypoint_indexes[image]
            inliersinimage=inliersindexes[start:end]
            keypoints=Keypoints[start:end,:]
            keypoints=keypoints[inliersinimage]

            image_descriptors=clustering.preprocess_features(Descriptors[start:end])
            image_descriptors=image_descriptors[inliersinimage]

            #calculate distance of each keypoints to each centroid
            distanceMatrix, clustering_assignments = KmeansClustering.index.search(image_descriptors, self.number_of_clusters)

            distanceMatrix=np.take_along_axis(distanceMatrix, np.argsort(clustering_assignments), axis=-1)

            #assign keypoints to centroids using the Hungarian algorithm. This ensures that each
            #image has at most one instance of each cluster
            keypointIndex,clusterAssignment= linear_sum_assignment(distanceMatrix)

            tempKeypoints=keypoints[keypointIndex]

            clusterAssignmentDistance = distanceMatrix[keypointIndex, clusterAssignment]

            clusterstokeep = np.zeros(len(clusterAssignmentDistance))
            clusterstokeep = clusterstokeep == 1

            # keep only points that lie in their below a cluster specific theshold
            clusterstokeep[clusterAssignmentDistance < thresholds[clusterAssignment]] = True

            tempKeypoints[:,2]=clusterAssignment

            Image_Keypoints[image]=tempKeypoints[clusterstokeep]
            averagepointsperimage+=sum(clusterstokeep)

        LogText(f"Keypoints Detected per image(clusteringAssignment) {averagepointsperimage / len(Image_Keypoints)}",self.experiment_name, self.log_path)
        ClosereadFileArray(fileHandler1,self.log_path+'CheckPoints/' +self.experiment_name + '/keypoints')
        ClosereadFileArray(fileHandler2,self.log_path+'CheckPoints/' +self.experiment_name + '/descriptors')
        self.save_keypoints(Image_Keypoints,"SuperPointKeypoints.pickle")
        LogText(f"Extraction of Initial pseudoGroundtruth completed", self.experiment_name, self.log_path)
        return Image_Keypoints


    def Indexes_of_inliers(self,Keypoints,Descriptors,buffersize):
        res = faiss.StandardGpuResources()
        nlist = 100
        quantizer = faiss.IndexFlatL2(256)
        index = faiss.IndexIVFFlat(quantizer, 256, nlist)

        gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index)

        gpu_index_flat.train(clustering.preprocess_features(Descriptors[:buffersize]))
        gpu_index_flat.add(clustering.preprocess_features(Descriptors[:buffersize]))

        rg = np.linspace(0, len(Descriptors), math.ceil(len(Descriptors) / 10000) + 1, dtype=int)
        keypoints_outlier_score=np.zeros(len(Keypoints))
        for i in range(len(rg) - 1):
            descr = clustering.preprocess_features(Descriptors[rg[i]:rg[i + 1], :])
            distance_to_closest_points, _ = gpu_index_flat.search(descr, 100)
            outlierscore = np.median(distance_to_closest_points, axis=1)
            keypoints_outlier_score[rg[i]:rg[i + 1]] = outlierscore

        inliers = keypoints_outlier_score.copy()
        inliers = np.sort(inliers)

        threshold = inliers[int((1-self.remove_superpoint_outliers_percentage) * (len(inliers) - 1))]
        inliers = keypoints_outlier_score < threshold
        return inliers


    def Indexes_of_BackgroundPoints(self,Keypoints,Descriptors,keypoint_indexes):
        backgroundpoitnsIndex = Keypoints[:, 2] == -1
        insideboxPoitnsIndex = Keypoints[:, 2] == 1

        backgroundDescriptors = clustering.preprocess_features(
            Descriptors[:500000 ][ [backgroundpoitnsIndex[:500000 ]]])

        insideboxDescriptors = clustering.preprocess_features(
            Descriptors[:500000][ [insideboxPoitnsIndex[:500000 ]]])

        number_of_insideClusters=100
        number_of_outsideClusters=250
        backgroundclustering= clustering.Kmeans(number_of_outsideClusters, centroids=None)
        insideboxclustering = clustering.Kmeans(number_of_insideClusters, centroids=None)

        backgroundclustering.cluster(backgroundDescriptors, verbose=False)
        insideboxclustering.cluster(insideboxDescriptors, verbose=False)

        foregroundpointindex=np.zeros(len(Keypoints))==-1
        for imagename in keypoint_indexes:
            start,end=keypoint_indexes[imagename]
            keypoints = Keypoints[start:end, :]
            descriptors=Descriptors[start:end,:]


            distanceinside, Iinside = insideboxclustering.index.search(clustering.preprocess_features(descriptors), 1)
            distanceoutside, Ioutside = backgroundclustering.index.search(clustering.preprocess_features(descriptors), 1)

            points_to_keep = (distanceinside < distanceoutside).reshape(-1)
            points_to_keep = np.logical_and(points_to_keep,keypoints[:,2]==1)
            foregroundpointindex[start:end] = points_to_keep

        return foregroundpointindex


    def GetPoints(self,confidenceMap, threshold, NMSthes):
        if(confidenceMap.size(1)==1):
            points,_=self.GetPointsFromHeatmap(confidenceMap, threshold, NMSthes)
            return points
        

        keypoints,keypointprob = self.GetPointsFromHeatmap(confidenceMap[:,0:1], threshold, NMSthes)
        keypoints1,keypoint1prob = self.GetPointsFromHeatmap(confidenceMap[:,1:2], threshold, NMSthes)
        keypoints2,keypoint2prob = self.GetPointsFromHeatmap(confidenceMap[:,2:3], threshold, NMSthes)
        

        keys = keypoints1
        imgaug_keypoints = []
        for j in range(len(keys)):
            imgaug_keypoints.append(imgaug.augmentables.kps.Keypoint(x=keys[j, 0], y=keys[j, 1]))
        kpsoi = imgaug.augmentables.kps.KeypointsOnImage(imgaug_keypoints, shape=confidenceMap.shape[2:])
        keypoitns_aug = self.SuperpointUndoScaleDistill1(keypoints=kpsoi)
        keys = keypoitns_aug.to_xy_array()
        keypoints1 = keys

        keys = keypoints2
        imgaug_keypoints = []
        for j in range(len(keys)):
            imgaug_keypoints.append(imgaug.augmentables.kps.Keypoint(x=keys[j, 0], y=keys[j, 1]))
        kpsoi = imgaug.augmentables.kps.KeypointsOnImage(imgaug_keypoints, shape=confidenceMap.shape[2:])
        keypoitns_aug = self.SuperpointUndoScaleDistill2(keypoints=kpsoi)
        keys = keypoitns_aug.to_xy_array()
        keypoints2 = keys

        newkeypoints = Cuda(torch.from_numpy(np.row_stack((keypoints.cpu().detach().numpy(),keypoints1,keypoints2))))
        newkeypointsprob = torch.cat((keypointprob,keypoint1prob,keypoint2prob))

        newkeypoints=torch.cat((newkeypoints,newkeypointsprob.unsqueeze(1)),1)
        newkeypoints = MergeScales(newkeypoints, int(NMSthes/2))                                                                                                                        

        return newkeypoints[:,:2]
            

    def GetPointsFromHeatmap(self,confidenceMap, threshold, NMSthes):
        mask = confidenceMap > threshold
        prob = confidenceMap[mask]
        value, indices = prob.sort(descending=True)
        pred = torch.nonzero(mask)
        prob = prob[indices]
        pred = pred[indices]
        points = pred[:, 2:4]
        points = points.flip(1)
        nmsPoints = torch.cat((points.float(), prob.unsqueeze(1)), 1).transpose(0, 1)
        thres = math.ceil(NMSthes / 2)  
        newpoints = torch.cat((nmsPoints[0:1, :] - thres, nmsPoints[1:2, :] - thres, nmsPoints[0:1, :] + thres,
                               nmsPoints[1:2, :] + thres, nmsPoints[2:3, :]), 0).transpose(0, 1)
        res = torchvision.ops.nms(newpoints[:, 0:4], newpoints[:, 4], 0.01)

        points = nmsPoints[:, res].transpose(0, 1)
        returnPoints = points[:, 0:2]
        prob = points[:, 2]
        return returnPoints,prob



    def GetSuperpointOutput(self,input):
        keypoints_volume, descriptors_volume = self.model(input)
        keypoints_volume = keypoints_volume.detach()
        keypoints_volume = self.softmax(keypoints_volume)
        volumeNoDustbin = keypoints_volume[:, :-1, :, :]
        spaceTensor = self.pixelSuffle(volumeNoDustbin)
        return spaceTensor,descriptors_volume



    def GetThresholdsPerCluster(self,inliersindexes,Descriptors,deepcluster):
        rg = np.linspace(0, sum(inliersindexes), math.ceil(sum(inliersindexes) / 10000) + 1, dtype=int)
        distance_to_centroid_per_cluster = list([[] for i in range(self.number_of_clusters)])

        for i in range(len(rg) - 1):
            descriptors = clustering.preprocess_features(Descriptors[rg[i]:rg[i + 1], :][inliersindexes[rg[i]:rg[i + 1]]])
            distancesFromCenter, clustering_assingments = deepcluster.index.search(descriptors, 1)
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



    def save_keypoints(self,Image_Keypoints,filename):
        checkPointdir = self.log_path+ 'CheckPoints/' + self.experiment_name + '/'
        checkPointFile=checkPointdir+filename
        with open(checkPointFile, 'wb') as handle:
            pickle.dump(Image_Keypoints, handle, protocol=pickle.HIGHEST_PROTOCOL)



# ----------------------------------------------------------------------    
#  https://github.com/magicleap/SuperPointPretrainedNetwork/
#
# --------------------------------------------------------------------*/
#

class SuperPointNet(torch.nn.Module):
  """ Pytorch definition of SuperPoint Network. """
  def __init__(self):
    super(SuperPointNet, self).__init__()
    self.relu = torch.nn.ReLU(inplace=True)
    self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
    self.numberOfClasses=1
    c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
    # Shared Encoder.
    self.conv1a = torch.nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
    self.conv1b = torch.nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
    self.conv2a = torch.nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
    self.conv2b = torch.nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
    self.conv3a = torch.nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
    self.conv3b = torch.nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
    self.conv4a = torch.nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
    self.conv4b = torch.nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)
    # Detector Head.
    self.convPa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
    self.convPb = torch.nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)
    # Descriptor Head.
    self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
    self.convDb = torch.nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)

  def forward(self, x):
    """ Forward pass that jointly computes unprocessed point and descriptor
    tensors.
    Input
      x: Image pytorch tensor shaped N x 1 x H x W.
    Output
      semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.git c
      desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
    """
    # Shared Encoder.
    x = self.relu(self.conv1a(x))
    x = self.relu(self.conv1b(x))
    x = self.pool(x)
    x = self.relu(self.conv2a(x))
    x = self.relu(self.conv2b(x))
    x = self.pool(x)
    x = self.relu(self.conv3a(x))
    x = self.relu(self.conv3b(x))
    x = self.pool(x)
    x = self.relu(self.conv4a(x))
    x = self.relu(self.conv4b(x))
    # Detector Head.
    cPa = self.relu(self.convPa(x))
    semi = self.convPb(cPa)
    # Descriptor Head.
    cDa = self.relu(self.convDa(x))
    desc = self.convDb(cDa)
    dn = torch.norm(desc, p=2, dim=1) # Compute the norm.
    desc = desc.div(torch.unsqueeze(dn, 1)) # Divide by norm to normalize.
    return semi, desc

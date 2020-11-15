from torch.utils.data import Dataset
from imgaug.augmentables.kps import Keypoint, KeypointsOnImage
import numpy as np
from Utils import *
import imgaug.augmenters as iaa
import imgaug.augmentables.kps 
import cv2
import random
import yaml

class Database(Dataset):
    def __init__(self,dataset_name,number_of_channels,test=False,image_keypoints=None, function_for_dataloading=None,augmentations=None,use_box=False):
        self.image_keypoints = image_keypoints
        self.number_of_channels=number_of_channels
        self.test=test
        self.use_box=use_box
        self.dataset_name=dataset_name
        self.preparedb()
        self.function_for_dataloading = function_for_dataloading
        self.augmentations=augmentations
        
        self.SuperpointScaleDistill1 = iaa.Affine(scale={"x": 1.3, "y": 1.3})
        self.SuperpointScaleDistill2 = iaa.Affine(scale={"x": 1.6, "y": 1.6})
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return self.function_for_dataloading(self,idx)


    def get_image_superpoint(self,idx):
        name = self.files[idx]

        image =self.getimage_superpoint(self,name)

        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_gray = torch.from_numpy(np.expand_dims(image_gray, 0) / 255.0).float()

        if(self.use_box):
            bbox=self.getbox(self,name)
            bbox = torch.tensor(bbox)
            sample={'image_gray': image_gray, 'filename': name,'bounding_box':bbox}
            return sample

        sample = {'image_gray': image_gray, 'filename': name}
        return sample



    def get_image_superpoint_multiple_scales(self,idx):
        name = self.files[idx]
        image =self.getimage_superpoint(self,name)

        image1 = self.SuperpointScaleDistill1(image=image)
        image2 = self.SuperpointScaleDistill2(image=image)

        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_gray = torch.from_numpy(np.expand_dims(image_gray, 0) / 255.0).float()

        image_gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        image_gray1 = torch.from_numpy(np.expand_dims(image_gray1, 0) / 255.0).float()

        image_gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        image_gray2 = torch.from_numpy(np.expand_dims(image_gray2, 0) / 255.0).float()

        imagegrayfinal = torch.cat((image_gray, image_gray1, image_gray2,), dim=0)
        
        if(self.use_box):
            bbox=self.getbox(self,name)
            bbox = torch.tensor(bbox)
            sample={'image_gray': imagegrayfinal, 'filename': name,'bounding_box':bbox}
            return sample

        sample = {'image_gray': imagegrayfinal, 'filename': name}

        return sample




    def get_FAN_inference(self,idx):
    
        name = self.files[idx]
        image =self.getimage_FAN(self,name)
        image =torch.from_numpy(image / 255.0).permute(2, 0, 1).float()

        sample = {'image': image, 'filename': name}
        return sample


    def get_FAN_secondStep_evaluation(self,idx):     
        name = self.files[idx]
        image =self.getimage_FAN(self,name)
        image =torch.from_numpy(image / 255.0).permute(2, 0, 1).float()
        is_it_test_sample=bool(self.is_test_sample[idx])
        groundtruth=torch.from_numpy(self.getGroundtruth(self,name,is_it_test_sample))
        sample = {'image': image, 'filename': name,'groundtruth':groundtruth,'is_it_test_sample':is_it_test_sample}
        return sample


    def get_FAN_secondStep_train(self, idx):        

        name = self.files[idx]
        keypoints = self.image_keypoints[name]
        image =self.getimage_FAN(self,name)


        imgaug_keypoints = []

        for i in range(len(keypoints)):
            imgaug_keypoints.append(Keypoint(x=4*keypoints[i, 0], y=4*keypoints[i, 1]))
        kpsoi = KeypointsOnImage(imgaug_keypoints, shape=image.shape)
        image, keypoitns_aug = self.augmentations(image=image, keypoints=kpsoi)

        keypoints = np.column_stack((keypoitns_aug.to_xy_array() / 4, keypoints[:, 2:]))


        image = torch.from_numpy(image / 255.0).permute(2, 0, 1).float()

        heatmaps_with_keypoints = torch.zeros(self.number_of_channels).bool()
        indeces = torch.from_numpy(keypoints[:, 2]).int().tolist()
        heatmaps_with_keypoints[indeces] = True
        shapegaussian = BuildMultiChannelGaussians(self.number_of_channels, keypoints.round())

        sample = {'image': image, 'GaussianShape': shapegaussian, 'heatmaps_with_keypoints': heatmaps_with_keypoints}

        return sample


    def get_FAN_firstStep_train(self,idx):

        imagesize=256
        heatmapsize=64
        keypointscaler=int(imagesize/heatmapsize)

        name1 = self.files[idx]
        keypoints1 = self.image_keypoints[name1]

        while(len(keypoints1)<9):
            idx = random.randint(0, len(self.files) - 1)
            name1 = self.files[idx]
            keypoints1 = self.image_keypoints[name1]

        image1 =self.getimage_FAN(self,name1)
        

        imgaug_keypoints = []

        for i in range(len(keypoints1)):
            imgaug_keypoints.append(Keypoint(x=keypointscaler*keypoints1[i, 0], y=keypointscaler*keypoints1[i, 1]))
        kpsoi = KeypointsOnImage(imgaug_keypoints, shape=image1.shape)
        image1, keypoitns_aug = self.augmentations(image=image1, keypoints=kpsoi)

        keypoints1 = np.column_stack((keypoitns_aug.to_xy_array()/keypointscaler, keypoints1[:, 2:]))
        keypoints1=keypoints1.round()

        image1 = torch.from_numpy(image1 / 255.0).permute(2, 0, 1).float()

        #sample a different image or use the same image with probability 50%
        if (random.random() <0.5):
            
            idx2 = random.randint(0, len(self.files) - 1)
            name2 = self.files[idx2]
            keypoints2 = self.image_keypoints[name2]
            while(len(keypoints2)<9):
                idx2 = random.randint(0, len(self.files) - 1)
                name2 = self.files[idx2]
                keypoints2 = self.image_keypoints[name2]

            image2 =self.getimage_FAN(self,name2)
            

        else:
            name2 = self.files[idx]
            image2 =self.getimage_FAN(self,name2)
            keypoints2 = self.image_keypoints[name2]


        imgaug_keypoints = []
        for i in range(len(keypoints2)):
            imgaug_keypoints.append(Keypoint(x=keypointscaler*keypoints2[i, 0], y=keypointscaler*keypoints2[i, 1]))
        kpsoi = KeypointsOnImage(imgaug_keypoints, shape=image2.shape)
        image2, keypoitns_aug = self.augmentations(image=image2, keypoints=kpsoi)

        keypoints2 = np.column_stack((keypoitns_aug.to_xy_array()/keypointscaler, keypoints2[:, 2:]))
        keypoints2=keypoints2.round()

        image2 = torch.from_numpy(image2 / 255.0).permute(2, 0, 1).float()

        image = torch.cat((image1, image2))

        number_of_pairs=3000
        pairs = -1*np.ones((number_of_pairs, 5))
        pair_index = 0

        # positive pairs
        for i in range(len(keypoints1)):
            if(keypoints1[i, 2]==-1):continue
            indxes = keypoints2[:, 2] == keypoints1[i, 2]
            coord1 = keypoints1[i, :2]
            coord2 = keypoints2[indxes, :2]

            if (len(coord2) == 0): continue

            coord2 = coord2[0]
            # check that not of the coordinates are out of range cause of the augmentations
            if (sum(coord1 > heatmapsize-1) == 0 and sum(coord1 < 0) == 0) and (sum(coord2 > heatmapsize-1) == 0 and sum(coord2 < 0) == 0):
                # if(np.random.rand(1)[0]<0.75):
                if (pair_index >= number_of_pairs - 1): break
                pairs[pair_index, :2] = coord1
                pairs[pair_index, 2:4] = coord2
                pairs[pair_index, 4] = 1.0
                pair_index += 1

        # negative pairs
        for i in range(len(keypoints1)):
            clust=keypoints1[i, 2]
            if(clust==-1 or ((clust in keypoints2[:,2]) is False) ):continue
            coord1 = keypoints1[i, :2]
            coord2s=keypoints2[keypoints2[:,2]!=clust]
            for j in range(len(coord2s)):
                clust2=keypoints2[j,2]
                if(clust2==-1 or (clust2 in keypoints1[:,2]) is False):continue
                coord2=coord2s[j,:2]
                if ((sum(coord1 > heatmapsize-2) == 0 and sum(coord1 < 0) == 0) and ( sum(coord2 > heatmapsize-2) == 0 and sum(coord2 < 0) == 0)):
                    if (pair_index >= number_of_pairs-1): break
                    pairs[pair_index, :2] = coord1
                    pairs[pair_index, 2:4] = coord2
                    pairs[pair_index, 4] = 0.0
                    pair_index += 1

        pairs=torch.from_numpy(pairs)
        gaussian1 = BuildGaussians(keypoints1)
        gaussian2 = BuildGaussians(keypoints2)

        gaussian=torch.cat((gaussian1.unsqueeze(0),gaussian2.unsqueeze(0)))

        sample = {'image': image, 'keypoints': pairs, 'keypointHeatmaps': gaussian}
        
        return sample




    def preparedb(self):
        if self.dataset_name == 'CelebA':
            
            def scaleforFAN(self,imagefile,keypoints):
                bbox = self.boxes[imagefile].copy()
                delta_x=1.2*bbox[2]-bbox[0]
                delta_y=2*bbox[3]-bbox[1]
                delta=0.5*(delta_x+delta_y)

                if(delta<20): tight_aux=8
                else: tight_aux=int(8*delta/100)

                minx=int(max(bbox[0]-tight_aux,0))
                miny=int(max(bbox[1]-tight_aux,0))
                maxx=int(min(bbox[2]+tight_aux,178-1))
                maxy=int(min(bbox[3]+tight_aux,218-1))
                
                keypoints[:,0]=keypoints[:,0]-minx
                keypoints[:,1]=keypoints[:,1]-miny
                keypoints[:,0]=keypoints[:,0]*(256/(maxx-minx))
                keypoints[:,1]=keypoints[:,1]*(256/(maxy-miny))
                return keypoints

            def getbox(self,imagefile):
                bbox = self.boxes[imagefile].copy()
                return bbox

            def getGroundtruth(self,imagefile,is_test_sample):
                groundtruthpoints=self.groundtruth[imagefile]
                groundtruthpoints=self.scaleforFAN(self,imagefile,groundtruthpoints)
                return groundtruthpoints

            def getimage_superpoint(self,imagefile):
                image = cv2.cvtColor(cv2.imread(self.datapath +imagefile), cv2.COLOR_BGR2RGB)
                return image

            def getimage_FAN(self,imagefile):
                image = cv2.cvtColor(cv2.imread(self.datapath +imagefile), cv2.COLOR_BGR2RGB)
                bbox = self.boxes[imagefile].copy()
                
                delta_x=1.2*bbox[2]-bbox[0]
                delta_y=2*bbox[3]-bbox[1]
                delta=0.5*(delta_x+delta_y)
                if(delta<20): tight_aux=8
                else: tight_aux=int(8*delta/100)

                minx=int(max(bbox[0]-tight_aux,0))
                miny=int(max(bbox[1]-tight_aux,0))
                maxx=int(min(bbox[2]+tight_aux,image.shape[1]-1))
                maxy=int(min(bbox[3]+tight_aux,image.shape[0]-1))

                image=image[miny:maxy,minx:maxx,:]
                image=cv2.resize(image,dsize=(256,256))
    
                return image


            self.getimage_superpoint=getimage_superpoint
            self.getimage_FAN=getimage_FAN
            self.getbox=getbox
            self.scaleforFAN=scaleforFAN
            self.getGroundtruth=getGroundtruth

            #load CelebA paths
            with open('paths/CelebA.yaml') as file:
                paths = yaml.load(file, Loader=yaml.FullLoader)
            self.datapath = paths['datapath']
            path_to_boxes= paths['path_to_boxes']
            self.boxes=load_keypoints(path_to_boxes)

            def init(self):
            
                #load CelebA paths
                with open('paths/CelebA.yaml') as file:
                    paths = yaml.load(file, Loader=yaml.FullLoader)

                list_eval_partition= paths['list_eval_partition']
                mafl_testing= paths['mafl_testing']
                mafl_training= paths['mafl_training']
                
                if (self.test):
                    with open(mafl_testing, 'r') as f:
                        TestImages = f.read().splitlines()
                    with open(mafl_training, 'r') as f:
                        TrainImages = f.read().splitlines()
                    

                    mafl_groundtruth= paths['mafl_groundtruth']
                    self.groundtruth=load_keypoints(mafl_groundtruth)

                    self.files = TrainImages[:1000] + TestImages
                    self.is_test_sample = np.ones(len(self.files))
                    self.is_test_sample[:1000]=0
                else:

                    self.boxes=load_keypoints(path_to_boxes)

                    with open(list_eval_partition, 'r') as f:
                        CelebAImages = f.read().splitlines()

                    CelebATrainImages=[f[:-2] for f in CelebAImages if f[-1]=='0']

                    with open(mafl_testing, 'r') as f:
                        MaflTestImages = f.read().splitlines()

                    CelebATrainImages=list(set(CelebATrainImages)-set(MaflTestImages))

                    self.files = CelebATrainImages

        if self.dataset_name == 'LS3D':
            
            def scaleforFAN(self,imagefile,keypoints):
                bbox = self.boxes[imagefile].copy()
                delta_x=1.2*bbox[2]-bbox[0]
                delta_y=2*bbox[3]-bbox[1]
                delta=0.5*(delta_x+delta_y)

                if(delta<20): tight_aux=8
                else: tight_aux=int(8*delta/100)

                minx=int(max(bbox[0]-tight_aux,0))
                miny=int(max(bbox[1]-tight_aux,0))
                maxx=int(min(bbox[2]+tight_aux,178-1))
                maxy=int(min(bbox[3]+tight_aux,218-1))
                
                keypoints[:,0]=keypoints[:,0]-minx
                keypoints[:,1]=keypoints[:,1]-miny
                keypoints[:,0]=keypoints[:,0]*(256/(maxx-minx))
                keypoints[:,1]=keypoints[:,1]*(256/(maxy-miny))
                return keypoints

            def getbox(self,imagefile):
                import torchfile
                keypoints=torchfile.load(imagefile[:-3]+'t7')
                bbox = self.boxes[imagefile].copy()
                return bbox

            def getGroundtruth(self,imagefile,is_test_sample):
                groundtruthpoints=self.groundtruth[imagefile]
                groundtruthpoints=self.scaleforFAN(self,imagefile,groundtruthpoints)
                return groundtruthpoints

            def getimage_superpoint(self,imagefile):
                image = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)

                import matplotlib
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(1)
                ax.set_axis_off()
                ax.imshow(image)
                # ax.scatter(keypoints[:, 0].cpu().detach().numpy(), keypoints[:, 1].cpu().detach().numpy())
                # ax.scatter(bounding_box[[0,2]].cpu().detach().numpy(), bounding_box[[1,3]].cpu().detach().numpy())
                plt.show()
                fig.savefig(f'/home/SERILOCAL/d.mallis/Projects/UnsupervisedLandmarks/foo.jpg')
                # fig.savefig(f'/home/SERILOCAL/d.mallis/Logs/test2/epoch15_{i}.jpg')
                

                return image

            def getimage_FAN(self,imagefile):
                image = cv2.cvtColor(cv2.imread(self.datapath +imagefile), cv2.COLOR_BGR2RGB)
                bbox = self.boxes[imagefile].copy()
                
                delta_x=1.2*bbox[2]-bbox[0]
                delta_y=2*bbox[3]-bbox[1]
                delta=0.5*(delta_x+delta_y)
                if(delta<20): tight_aux=8
                else: tight_aux=int(8*delta/100)

                minx=int(max(bbox[0]-tight_aux,0))
                miny=int(max(bbox[1]-tight_aux,0))
                maxx=int(min(bbox[2]+tight_aux,image.shape[1]-1))
                maxy=int(min(bbox[3]+tight_aux,image.shape[0]-1))

                image=image[miny:maxy,minx:maxx,:]
                image=cv2.resize(image,dsize=(256,256))
    
                return image


            self.getimage_superpoint=getimage_superpoint
            self.getimage_FAN=getimage_FAN
            self.getbox=getbox
            self.scaleforFAN=scaleforFAN
            self.getGroundtruth=getGroundtruth

            with open('paths/LS3D.yaml') as file:
                paths = yaml.load(file, Loader=yaml.FullLoader)
            self.datapath = paths['datapath']

        
            def init(self):            
                if (self.test):
                    with open(mafl_testing, 'r') as f:
                        TestImages = f.read().splitlines()
                    with open(mafl_training, 'r') as f:
                        TrainImages = f.read().splitlines()
                    
                    mafl_groundtruth= paths['mafl_groundtruth']
                    self.groundtruth=load_keypoints(mafl_groundtruth)

                    self.files = TrainImages[:1000] + TestImages
                    self.is_test_sample = np.ones(len(self.files))
                    self.is_test_sample[:1000]=0
                else:
                    self.files = glob.glob(self.datapath + '/**/*.jpg', recursive=True)+glob.glob(self.datapath + '/**/*.png', recursive=True)

        if (self.image_keypoints is not None):
            self.files = list(self.image_keypoints.keys())
        else:
            init(self)

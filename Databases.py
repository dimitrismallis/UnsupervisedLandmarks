from torch.utils.data import Dataset
from imgaug.augmentables.kps import Keypoint, KeypointsOnImage
import numpy as np
from Utils import *
import imgaug.augmenters as iaa
import imgaug.augmentables.kps 
import cv2
import random
import yaml
import torchfile
import scipy.io
from pathlib import Path

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
        is_it_test_sample=bool(self.is_test_sample[idx])
        image =self.getimage_FAN(self,name,is_it_test_sample=is_it_test_sample)
        image =torch.from_numpy(image / 255.0).permute(2, 0, 1).float()
        
        groundtruth=torch.from_numpy(self.getGroundtruth(self,name,is_it_test_sample))
        sample = {'image': image, 'filename': name,'groundtruth':groundtruth,'is_it_test_sample':is_it_test_sample}
        return sample


    def get_FAN_secondStep_train(self, idx):        

        name = self.files[idx]
        keypoints = self.image_keypoints[name]
        image,keypoints =self.getimage_FAN(self,name,self.augmentations, 4 * keypoints)


        keypoints = keypoints/4
        keypoints=keypoints.round()

        image = torch.from_numpy(image / 255.0).permute(2, 0, 1).float()
        heatmaps_with_keypoints = torch.zeros(self.number_of_channels).bool()
        indeces = torch.from_numpy(keypoints[:, 2]).int().tolist()
        heatmaps_with_keypoints[indeces] = True
        shapegaussian = BuildMultiChannelGaussians(self.number_of_channels, keypoints.round())

        sample = {'image': image, 'GaussianShape': shapegaussian, 'heatmaps_with_keypoints': heatmaps_with_keypoints}
        return sample


    def get_FAN_firstStep_train(self, idx):     
        heatmapsize=64
        name1 = self.files[idx]
        keypoints1 = self.image_keypoints[name1]

        while(len(keypoints1)<9):
            idx = random.randint(0, len(self.files) - 1)
            name1 = self.files[idx]
            keypoints1 = self.image_keypoints[name1]

        image1 ,keypoints1=self.getimage_FAN(self,name1,self.augmentations,4*keypoints1)
        
        keypoints1 = keypoints1/4
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

            image2 ,keypoints2=self.getimage_FAN(self,name2,self.augmentations,4*keypoints2)
            

        else:
            name2 = self.files[idx]
            keypoints2 = self.image_keypoints[name2]
            image2 ,keypoints2=self.getimage_FAN(self,name2,self.augmentations,4*keypoints2)

        keypoints2=keypoints2.round()
        keypoints2 = keypoints2/4
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

        def GetFullImagePath(self,imagefile,istestsample=False):
            if self.dataset_name =='CelebA': 
                    return self.datapath+imagefile

            if self.dataset_name =='Human3.6': 
                return self.imagepath+imagefile
            
            if self.dataset_name =='LS3D':
                if(istestsample):
                    return  imagefile
                return self.datapath+imagefile



        if self.dataset_name in ['CelebA','LS3D']:
            
            def getFANBox(self,imagefile,W,H,is_test_sample=False):
                bbox = self.getbox(self,imagefile,is_test_sample)
                delta_x=1.2*bbox[2]-bbox[0]
                delta_y=2*bbox[3]-bbox[1]
                delta=0.5*(delta_x+delta_y)

                if(delta<20): tight_aux=8
                else: tight_aux=int(8*delta/100)

                minx=int(max(bbox[0]-tight_aux,0))
                miny=int(max(bbox[1]-tight_aux,0))
                maxx=int(min(bbox[2]+tight_aux,W-1))
                maxy=int(min(bbox[3]+tight_aux,H-1))

                return minx,miny,maxx,maxy

            def keypointsToFANResolution(self,imagefile,keypoints,W=None,H=None,is_test_sample=False):
                
                if(W is None or H is None):
                    W=self.W
                    H=self.H
                minx,miny,maxx,maxy=self.getFANBox(self,imagefile,W,H,is_test_sample)

                keypoints[:,0]=keypoints[:,0]-minx
                keypoints[:,1]=keypoints[:,1]-miny
                keypoints[:,0]=keypoints[:,0]*(256/(maxx-minx))
                keypoints[:,1]=keypoints[:,1]*(256/(maxy-miny))
                return keypoints


            def keypointsToOriginalResolution(self,imagefile,keypoints):

                minx,miny,maxx,maxy=self.getFANBox(self,imagefile,self.W,self.H)
                
                keypoints[:,0]=keypoints[:,0]*((maxx-minx)/256)
                keypoints[:,1]=keypoints[:,1]*((maxy-miny)/256)
                keypoints[:,0]=keypoints[:,0]+minx
                keypoints[:,1]=keypoints[:,1]+miny
                return keypoints

            def getbox(self,imagefile,is_test_sample=False):
                bbox = self.boxes[imagefile].copy()
                return bbox
            
            def getbox_fromlandmarks_ls3d_eval(self,imagefile,is_test_sample=False):
                
                try:
                    if(is_test_sample):  
                        gt=torchfile.load(imagefile[:-4]+'.t7')
                    else:
                        gt_filename=self.GetFullImagePath(self,imagefile,is_test_sample)[:-4]+'.t7'
                        tempstring=gt_filename.split('/')
                        tempstring.insert(-2,'landmarks')
                        tempstring='/'.join(tempstring)
                        gt_filename=tempstring[:-3]+'_pts.mat'
                        gt=scipy.io.loadmat(gt_filename)['pts_3d']
                except:
                   pass         
            
                
                bbox=[0,0,0,0]
                bbox[0]=int(min(gt[:,0]))
                bbox[1]=int(min(gt[:,1]))
                bbox[2]=int(max(gt[:,0]))
                bbox[3]=int(max(gt[:,1]))

                bbox[1]=bbox[1]-(bbox[3]-bbox[1])/3
                return bbox


            def getGroundtruth_MALF(self,imagefile,is_test_sample):
                groundtruthpoints=self.groundtruth[imagefile]
                groundtruthpoints=self.keypointsToFANResolution(self,imagefile,groundtruthpoints,self.W,self.H)
                return groundtruthpoints
            

            def getGroundtruth_LS3D(self,imagefile,is_test_sample):
                image = cv2.cvtColor(cv2.imread(self.GetFullImagePath(self,imagefile,is_test_sample)), cv2.COLOR_BGR2RGB)
                if(is_test_sample):  
                    groundtruthpoints=torchfile.load(imagefile[:-4]+'.t7')
                else:
                    gt_filename=self.GetFullImagePath(self,imagefile,is_test_sample)[:-4]+'.t7'
                    tempstring=gt_filename.split('/')
                    tempstring.insert(-2,'landmarks')
                    tempstring='/'.join(tempstring)
                    gt_filename=tempstring[:-3]+'_pts.mat'
                    groundtruthpoints=scipy.io.loadmat(gt_filename)['pts_3d']
                
                groundtruthpoints=self.keypointsToFANResolution(self,imagefile,groundtruthpoints,image.shape[1],image.shape[0],is_test_sample)
                return groundtruthpoints

            def getimage_superpoint(self,imagefile):
                image = cv2.cvtColor(cv2.imread(self.GetFullImagePath(self,imagefile,False)), cv2.COLOR_BGR2RGB)
                return image


            def getimage_FAN(self,imagefile, augmentations=None, keypoints=None,is_it_test_sample=False):

                image = cv2.cvtColor(cv2.imread(self.GetFullImagePath(self,imagefile,is_it_test_sample)), cv2.COLOR_BGR2RGB)
                
                if(augmentations is not None):
                    keypoints_originalres=self.keypointsToOriginalResolution(self,imagefile,keypoints)
                    imgaug_keypoints = []
                    for i in range(len(keypoints)):
                        imgaug_keypoints.append(Keypoint(x=keypoints_originalres[i, 0], y=keypoints_originalres[i, 1]))
                    kpsoi = KeypointsOnImage(imgaug_keypoints, shape=image.shape)
                    image, keypoitns_aug = self.augmentations(image=image, keypoints=kpsoi)

                    keypoints_originalres = np.column_stack((keypoitns_aug.to_xy_array(), keypoints_originalres[:, 2:]))


                minx,miny,maxx,maxy=self.getFANBox(self,imagefile,image.shape[1],image.shape[0],is_it_test_sample)

                image=image[miny:maxy,minx:maxx,:]
                image=cv2.resize(image,dsize=(256,256))

                if(keypoints is not None):
                    augmentedkeypoints=self.keypointsToFANResolution(self,imagefile,keypoints_originalres,self.W,self.H)


                    return image,augmentedkeypoints
                
                return image

            self.GetFullImagePath=GetFullImagePath
            self.keypointsToOriginalResolution=keypointsToOriginalResolution
            self.keypointsToFANResolution=keypointsToFANResolution
            self.getimage_superpoint=getimage_superpoint
            self.getimage_FAN=getimage_FAN
            self.getbox=getbox
            self.getFANBox=getFANBox
            


        if self.dataset_name == 'CelebA':
            #load CelebA paths
            with open('paths/main.yaml') as file:
                paths = yaml.load(file, Loader=yaml.FullLoader)
            

            self.datapath = paths['CelebA_datapath']

            assert self.datapath!=None, "Path missing!! Update 'CelebA_datapath' on paths/main.yaml with path to CelebA images."
            assert Path(self.datapath).exists(), f'Specified path to CelebA images does not exists {self.datapath}'


            with open('data/CelebA/list_eval_partition.txt', 'r') as f:
                CelebAImages = f.read().splitlines()
            assert len(list(Path(self.datapath).glob('*.jpg')))==len(CelebAImages), f"There are missing CelebA images from {self.datapath}. Please specify a path that includes all CelebA images"
            
            self.boxes=load_keypoints('data/CelebA/CelebABoundingBoxes.pickle')
            self.H=218
            self.W=178
            
            def init(self):           
                if (self.test):
                    with open('data/CelebA/mafl_testing.txt', 'r') as f:
                        TestImages = f.read().splitlines()
                    with open('data/CelebA/mafl_training.txt', 'r') as f:
                        TrainImages = f.read().splitlines()
                    
                    self.groundtruth=load_keypoints('data/CelebA/MaflGroundtruthLandmarks.pickle')
                    self.files = TrainImages[:1000] + TestImages
                    self.is_test_sample = np.ones(len(self.files))
                    self.is_test_sample[:1000]=0
                    self.getGroundtruth=getGroundtruth_MALF
                else:

                    with open('data/CelebA/list_eval_partition.txt', 'r') as f:
                        CelebAImages = f.read().splitlines()

                    CelebATrainImages=[f[:-2] for f in CelebAImages if f[-1]=='0']
                    with open('data/CelebA/mafl_testing.txt', 'r') as f:
                        MaflTestImages = f.read().splitlines()

                    CelebATrainImages=list(set(CelebATrainImages)-set(MaflTestImages))
                    self.files = CelebATrainImages

        if self.dataset_name == 'LS3D':
            

            self.boxes=load_keypoints('data/LS3D/300W_LPBoundingBoxes.pickle')

            with open('paths/main.yaml') as file:
                paths = yaml.load(file, Loader=yaml.FullLoader)
            self.datapath = paths['300WLP_datapath']

            assert self.datapath!=None, "Path missing!! Update '300WLP_datapath' on paths/main.yaml with path to 300WLP images."


            self.path_to_LS3Dbalanced=paths['LS3Dbalanced_datapath']

            self.H=450
            self.W=450   

            def init(self):
                         
                if (self.test):

                    assert self.path_to_LS3Dbalanced!=None, "Path missing!! Update 'LS3Dbalanced_datapath' on paths/main.yaml with path to LS3Dbalanced images."

                    self.getbox=getbox_fromlandmarks_ls3d_eval
                    self.getGroundtruth=getGroundtruth_LS3D
                    testfiles = glob.glob(self.path_to_LS3Dbalanced + '/**/*.jpg', recursive=True)
                    trainfiles= list(self.boxes.keys())
                   
                    self.files=trainfiles[:1000]+testfiles
                    self.is_test_sample = np.ones(len(self.files))
                    self.is_test_sample[:1000]=0
                else:
                    self.files = list(self.boxes.keys())

        if self.dataset_name == 'Human3.6':
            
            def tranformKeypoints(self,keypoints,augmentation,imageshape):
                imgaug_keypoints = []
                for i in range(len(keypoints)):
                    imgaug_keypoints.append(Keypoint(x=keypoints[i, 0], y=keypoints[i, 1]))
                kpsoi = KeypointsOnImage(imgaug_keypoints, shape=imageshape)
                keypoitns_aug = augmentation(keypoints=kpsoi)
                if(isinstance(keypoints,np.ndarray)):
                    keypoints[:,:2] = keypoitns_aug.to_xy_array()
                else:
                    keypoints[:,:2] = torch.from_numpy(keypoitns_aug.to_xy_array())
                return keypoints

            def keypointsToFANResolution(self,imagefile,keypoints):
                return self.tranformKeypoints(self,keypoints,self.scaleToFANRes,(450,450))

            def keypointsToOriginalResolution(self,imagefile,keypoints):
                return self.tranformKeypoints(self,keypoints,self.scaleToOriginalRes,(256,256))

            def getbox(self,imagefile,is_test_sample=False):
                bbox = self.boxes[imagefile].copy()
                return bbox

            def getGroundtruth(self,imagefile,is_test_sample):
                groundtruthpoints=self.groundtruth[imagefile]
                groundtruthpoints=self.flipGroundtruths(self,groundtruthpoints)
                groundtruthpoints=self.keypointsToFANResolution(self,imagefile,groundtruthpoints)
                return groundtruthpoints

            def getimage_superpoint(self,imagefile):
                image = cv2.cvtColor(cv2.imread(self.GetFullImagePath(self,imagefile,False)), cv2.COLOR_BGR2RGB)
                return image

            def getimage_FAN(self,imagefile, augmentations=None, keypoints=None,is_it_test_sample=False):

                image = cv2.cvtColor(cv2.imread(self.GetFullImagePath(self,imagefile,is_it_test_sample)), cv2.COLOR_BGR2RGB)
                
                if(augmentations is not None):
                    keypoints_originalres=self.keypointsToOriginalResolution(self,imagefile,keypoints)
                    imgaug_keypoints = []
                    for i in range(len(keypoints)):
                        imgaug_keypoints.append(Keypoint(x=keypoints_originalres[i, 0], y=keypoints_originalres[i, 1]))
                    kpsoi = KeypointsOnImage(imgaug_keypoints, shape=image.shape)
                    image, keypoitns_aug = self.augmentations(image=image, keypoints=kpsoi)

                    keypoints_originalres = np.column_stack((keypoitns_aug.to_xy_array(), keypoints_originalres[:, 2:]))


                scaledImage=self.scaleToFANRes(image=image)
                
                if(keypoints is not None):
                    augmentedkeypoints=self.keypointsToFANResolution(self,imagefile,keypoints_originalres)




                    return scaledImage,augmentedkeypoints
                return scaledImage
            
            def flipGroundtruths(self,keypoints):
                keypoints = np.concatenate( (keypoints,np.expand_dims(np.array(range(len(keypoints))), axis=1)), axis=1)
                matchedPart1 = np.array( [[1, 6],  [25, 17], [18, 26], [27, 19], [20, 28], [29, 21], [30, 22], [31, 23],  ])
                matchedPart2 = np.array( [[2, 7],[3, 8], [4, 9], [5, 10]])

                if (keypoints[1, 0]  >keypoints[6, 0]):
                    for i in range(matchedPart1.shape[0]):
                        idx1, idx2 = matchedPart1[i]
                        temp = keypoints[idx1,2]
                        keypoints[idx1,2] = keypoints[idx2,2]
                        keypoints[idx2,2] = temp

                if (keypoints[2, 0] > keypoints[7, 0]):
                    for i in range(matchedPart2.shape[0]):
                        idx1, idx2 = matchedPart2[i]
                        temp = keypoints[idx1, 2]
                        keypoints[idx1, 2] = keypoints[idx2, 2]
                        keypoints[idx2, 2] = temp
                
                keypoints=keypoints[np.argsort(keypoints[:,2])]
                keypoints=keypoints[:,:2]
                return keypoints


            self.flipGroundtruths=flipGroundtruths
            self.GetFullImagePath=GetFullImagePath
            self.tranformKeypoints=tranformKeypoints
            self.keypointsToOriginalResolution=keypointsToOriginalResolution
            self.keypointsToFANResolution=keypointsToFANResolution
            self.getimage_superpoint=getimage_superpoint
            self.getimage_FAN=getimage_FAN
            self.getbox=getbox
            self.getGroundtruth=getGroundtruth

            with open('paths/main.yaml') as file:
                paths = yaml.load(file, Loader=yaml.FullLoader)
            self.datapath = paths['Human_datapath']
            self.imagepath=self.datapath + 'images'

            try:
                self.boxes=load_keypoints(self.datapath+'HumanBoundingBoxes.pickle')
            except:
                filename=self.datapath+'HumanBoundingBoxes.pickle'
                raise Exception('File '+filename+' was not found ')

            try:
                self.groundtruth=load_keypoints(self.datapath+'GroundtruthKeypoints.pickle')
            except:
                filename=self.datapath+'GroundtruthKeypoints.pickle'
                raise Exception('File '+filename+' was not found ')


            self.scaleToFANRes = iaa.Sequential([iaa.Affine(scale={"x": 1.4, "y": 1.4}), iaa.Resize(256)])
            self.scaleToOriginalRes = iaa.Sequential([iaa.Resize(450), iaa.Affine(scale={"x": 1/1.4, "y": 1/1.4})])
            def init(self):
                self.files = list(k for k in self.boxes.keys())
                if (self.test):
                    
                    filestrain=[f for f in self.files if 'train' in f][::50]
                    filestest=[f for f in self.files if 'test' in f]

                    self.files=filestrain[:1000]+filestest
                    self.is_test_sample = np.ones(len(self.files))
                    self.is_test_sample[:1000]=0
                else:
                    self.files=[f for f in self.files if 'train' in f]
        if (self.image_keypoints is not None):
            self.files = list(self.image_keypoints.keys())
        else:
            init(self)

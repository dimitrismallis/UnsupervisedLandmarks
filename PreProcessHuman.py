import cv2
import numpy as np
import scipy.io as sio
import h5py
import cdflib
import glob
import os
import pickle
import numpy as np
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path_to_extract_dataset', help="Select path to which the dataset is going to be extracted")
parser.add_argument('--path_to_Human',  help="Path where Human3.6 database is downloaded to. Example <your_path>/training")
args=parser.parse_args()


def Create_Dataset(datapath,pathToHuman):

    pathToImages=datapath+'images'

    if not os.path.exists(pathToImages):
        os.makedirs(pathToImages)

    if not os.path.exists(pathToImages+'/train'):
        os.makedirs(pathToImages+'/train')

    if not os.path.exists(pathToImages+'/test'):
        os.makedirs(pathToImages+'/test')

    files = glob.glob(pathToHuman + '/**/*.mp4', recursive=True)

    trainVideos=[f for f in files if 'S1/' in f or 'S5/' in f or 'S6/' in f or 'S7/' in f or 'S8/' in f or 'S9/' in f ]
    trainVideos=[f for f in trainVideos if 'Directions' in f or 'Discussion' in f or 'Posing' in f or 'Waiting' in f or 'Greeting' in f or 'Walking' in f ]

    testVideos=[f for f in files if 'S11/' in f]
    testVideos=[f for f in testVideos if 'Directions' in f or 'Discussion' in f or 'Posing' in f or 'Waiting' in f or 'Greeting' in f or 'Walking' in f ]

    BoundingBoxes={}
    Keypoints={}

    imagecount=0
    for videofile in testVideos:
        boxfile=videofile.replace('/Videos','/MySegmentsMat/ground_truth_bb')[:-4]+'.mat'
        keypointfile=videofile.replace('/Videos','/MyPoseFeatures/D2_Positions')[:-4]+'.cdf'

        vid=cv2.VideoCapture(videofile)
        boxes=h5py.File(boxfile)
        points= cdflib.CDF(keypointfile)
        vid_length = boxes['Masks'].shape[0]

        timedistance=50
        for i in range(vid_length):
            success,image = vid.read()
            if(i % timedistance==0):
                
                image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                framebox=boxes[boxes['Masks'][i][0]][:].T
                whereres=np.where(framebox==1)
                minx=min(whereres[1])
                miny=min(whereres[0])
                maxx=max(whereres[1])
                maxy=max(whereres[0])
                framebox=[minx, miny, maxx, maxy]
                framekeypoint= points['Pose'][0,i,:].reshape(32,2)

                imagename='/test'+'/'+str(imagecount).zfill(6)+'.jpg'



                c = [(framebox[0] + framebox[2]) / 2, (framebox[1] + framebox[3]) / 2]
                s = 1.5 * (framebox[3] - framebox[1]) / 200
                r = 0
                trans = get_affine_transform(np.array(c), s, r, np.array([450, 450]))

                image_cropped = cv2.warpAffine(
                    image,
                    trans,
                    (450, 450),
                    flags=cv2.INTER_LINEAR)

                bbox_cropped = framebox.copy()
                tr = affine_transform(np.array([bbox_cropped[0], bbox_cropped[1]]), trans)
                bbox_cropped[0] = tr[0]
                bbox_cropped[1] = tr[1]
                tr = affine_transform(np.array([bbox_cropped[2], bbox_cropped[3]]), trans)
                bbox_cropped[2] = tr[0]
                bbox_cropped[3] = tr[1]

                joints_cropped = framekeypoint.copy()
                for j in range(len(joints_cropped)):
                    joints_cropped[j, :] = affine_transform(joints_cropped[j, :], trans)
                framekeypoint=joints_cropped
                framebox=bbox_cropped

                image=image_cropped
                

                BoundingBoxes[imagename]=framebox
                Keypoints[imagename]=framekeypoint

                cv2.imwrite(pathToImages+imagename,image[:,:,[2,1,0]])
                imagecount+=1


    imagecount=0
    for videofile in trainVideos:

        boxfile=videofile.replace('/Videos','/MySegmentsMat/ground_truth_bb')[:-4]+'.mat'
        keypointfile=videofile.replace('/Videos','/MyPoseFeatures/D2_Positions')[:-4]+'.cdf'

        vid=cv2.VideoCapture(videofile)
        boxes=h5py.File(boxfile)
        points= cdflib.CDF(keypointfile)
        vid_length = boxes['Masks'].shape[0]

        timedistance=10
        for i in range(vid_length):

            success,image = vid.read()
            if(i % timedistance==0):

                image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                framebox=boxes[boxes['Masks'][i][0]][:].T
                whereres=np.where(framebox==1)
                minx=min(whereres[1])
                miny=min(whereres[0])
                maxx=max(whereres[1])
                maxy=max(whereres[0])
                framebox=[minx, miny, maxx, maxy]
                framekeypoint= points['Pose'][0,i,:].reshape(32,2)

                imagename='/train'+'/'+str(imagecount).zfill(6)+'.jpg'
                

                c = [(framebox[0] + framebox[2]) / 2, (framebox[1] + framebox[3]) / 2]
                s = 1.5 * (framebox[3] - framebox[1]) / 200
                r = 0
                trans = get_affine_transform(np.array(c), s, r, np.array([450, 450]))

                image_cropped = cv2.warpAffine(
                    image,
                    trans,
                    (450, 450),
                    flags=cv2.INTER_LINEAR)


                bbox_cropped = framebox.copy()
                tr = affine_transform(np.array([bbox_cropped[0], bbox_cropped[1]]), trans)
                bbox_cropped[0] = tr[0]
                bbox_cropped[1] = tr[1]
                tr = affine_transform(np.array([bbox_cropped[2], bbox_cropped[3]]), trans)
                bbox_cropped[2] = tr[0]
                bbox_cropped[3] = tr[1]

                joints_cropped = framekeypoint.copy()
                for j in range(len(joints_cropped)):
                    joints_cropped[j, :] = affine_transform(joints_cropped[j, :], trans)
                framekeypoint=joints_cropped


                framebox=bbox_cropped
                BoundingBoxes[imagename]=framebox
                image=image_cropped
                Keypoints[imagename]=framekeypoint

                cv2.imwrite(pathToImages+imagename,image[:,:,[2,1,0]])
                imagecount+=1



    boundingBoxFile=datapath+'/HumanBoundingBoxes.pickle'
    with open(boundingBoxFile,'wb') as handle:
        pickle.dump(BoundingBoxes,handle,protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Bournding Boxes saved at :{boundingBoxFile}")

    GroundtruthKeypointsFile=datapath+'/GroundtruthKeypoints.pickle'
    with open(GroundtruthKeypointsFile,'wb') as handle:
        pickle.dump(Keypoints,handle,protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Keypoints for evaluation saved at :{GroundtruthKeypointsFile}")



def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale])

    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


datapath=args.path_to_extract_dataset
pathToHuman=args.path_to_Human
Create_Dataset(datapath,pathToHuman)
print('Dataset extraction completed')

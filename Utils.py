import torch
import tables
import os
import pickle
import numpy as np
import math
import datetime
import torchvision
import cv2
import glob

def LogText(text,Experiment_name,log_path):
    print(text + "  (" + datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y") + ")")


    log_File = log_path + 'Logs/' + Experiment_name + '/' + Experiment_name + '.txt'

    f = open(log_File, 'a')
    f.write(text + "  (" + datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y") + ")" + '\n')
    f.close()


def GetPathsFromExperiment(experiment_name,log_path):
    CheckPointDirectory = log_path+'CheckPoints/' +experiment_name + '/*'
    listoffiles=glob.glob(CheckPointDirectory)

    path_to_checkpoint=max([f for f in listoffiles if '.pth' in f and 'FirstStep' in f ], key=os.path.getctime)
    path_to_keypoints=max([f for f in listoffiles if '.pickle' in f and 'UpdatedKeypoints' in f], key=os.path.getctime)

    return path_to_checkpoint,path_to_keypoints

def GetPathsResumeFirstStep(experiment_name,log_path):
    CheckPointDirectory = log_path+'CheckPoints/' +experiment_name + '/*'
    listoffiles=glob.glob(CheckPointDirectory)

    path_to_keypoints=max([f for f in listoffiles if '.pickle' in f and ('SuperPoint' in f or 'Updated' in f) ] , key=os.path.getctime)

    sortedlistoffiles=sorted(listoffiles,key=os.path.getctime)

    indexofcheckpoint=sortedlistoffiles.index(path_to_keypoints)
    sortedlistoffiles=sortedlistoffiles[:indexofcheckpoint]

    path_to_checkpoint=max([f for f in sortedlistoffiles if '.pth' in f and 'FirstStep' in f ], key=os.path.getctime)

    return path_to_checkpoint,path_to_keypoints

def GetPathsResumeSecondStep(experiment_name,log_path):
    CheckPointDirectory = log_path+'CheckPoints/' +experiment_name + '/*'
    listoffiles=glob.glob(CheckPointDirectory)

    sortedlistoffiles=sorted(listoffiles,key=os.path.getctime)

    path_to_checkpoint=max([f for f in listoffiles if '.pth' in f and 'SecondStep' in f], key=os.path.getctime)

    indexofcheckpoint=sortedlistoffiles.index(path_to_checkpoint)
    sortedlistoffiles=sortedlistoffiles[:indexofcheckpoint]

    path_to_keypoints=max([f for f in sortedlistoffiles if '.pickle' in f and ('Merged' in f or 'Updated' in f) ] , key=os.path.getctime)

    return path_to_checkpoint,path_to_keypoints

def GetPathsEval(experiment_name,log_path):
    CheckPointDirectory = log_path+'CheckPoints/' +experiment_name + '/*'
    listoffiles=glob.glob(CheckPointDirectory)

    path_to_checkpoint=max([f for f in listoffiles if '.pth' in f and 'SecondStep' in f], key=os.path.getctime)

    return path_to_checkpoint

def initialize_log_dirs(Experiment_name,log_path):
    CheckPointDirectory = log_path+'CheckPoints/' +Experiment_name + '/'
    Log_directory = log_path+'Logs/' + Experiment_name + '/'


    if not os.path.exists(Log_directory):
        os.makedirs(Log_directory)

    if not os.path.exists(CheckPointDirectory):
        os.makedirs(CheckPointDirectory)


def load_keypoints(filename):
    checkPointFile=filename
    with open(checkPointFile, 'rb') as handle:
        Image_Keypoints=pickle.load( handle)
    return Image_Keypoints

def Cuda(model):
    if torch.cuda.is_available():
        return model.cuda()
    else:
        return model


def CreateFileArray(name,columns):
    filename = name+f'.npy'
    if os.path.exists(filename):
        os.remove(filename)
    f = tables.open_file(filename, mode='w')
    atom = tables.Float64Atom()
    f.create_earray(f.root, 'data', atom, (0, columns))
    f.close()

def AppendFileArray(array,name):
    filename = name+f'.npy'
    f = tables.open_file(filename, mode='a')
    f.root.data.append(array)
    f.close()


def OpenreadFileArray(name):
    filename = name+f'.npy'
    f = tables.open_file(filename, mode='r')
    a=f.root.data
    return a,f

def ClosereadFileArray(f,name):
    filename = name+f'.npy'
    f.close()
    if os.path.exists(filename):
        os.remove(filename)


def load_keypoints(filename):
    with open(filename, 'rb') as handle:
        Image_Keypoints=pickle.load( handle)
    return Image_Keypoints


def save_keypoints(Image_Keypoints,filename,experiment):
    checkPointFile=checkPointdir+filename
    with open(checkPointFile, 'wb') as handle:
        pickle.dump(Image_Keypoints, handle, protocol=pickle.HIGHEST_PROTOCOL)


def BuildMultiChannelGaussians(outputChannels,keypoints,resolution=64,size=3):
    points = keypoints.copy()
    points[:, 0] = points[:, 0] + 1
    points[:, 1] = points[:, 1] + 1

    numberOfAnnotationsPoints = points.shape[0]
    heatMaps=torch.zeros(outputChannels, resolution, resolution)

    for i in range(numberOfAnnotationsPoints):
        p=np.asarray(points[i]) 
        try:
            heatMaps[int(p[2])] = fastDrawGaussian(heatMaps[int(p[2])], p, size)                                                                                                    
        except:
            pass

    return heatMaps


def MergeScales(points,thres):

    nmsPoints=points.T

    newpoints = torch.cat((nmsPoints[0:1, :] - thres, nmsPoints[1:2, :] - thres, nmsPoints[0:1, :] + thres,
                           nmsPoints[1:2, :] + thres, nmsPoints[2:3, :]), 0).T
    res = torchvision.ops.nms(newpoints[:, 0:4], newpoints[:, 4], 0.01)

    points = nmsPoints[:, res].T
    return points


def BuildGaussians(keypoints,resolution=64,size=1):
    points = keypoints.copy()
    points[:, 0] = points[:, 0] + 1
    points[:, 1] = points[:, 1] + 1


    numberOfAnnotationsPoints = points.shape[0]
    if (numberOfAnnotationsPoints == 0):
        heatMaps=torch.zeros(1, resolution, resolution)
    else:
        heatMaps = torch.zeros(numberOfAnnotationsPoints, resolution, resolution)
    for i in range(numberOfAnnotationsPoints):
        p=np.asarray(points[i])
        try:
            heatMaps[i] = fastDrawGaussian(heatMaps[i], p, size)

        except:
            pass
    heatmap = torch.max(heatMaps, 0)[0]
    return heatmap



def MergePoints(current_points,oldPoints):
    current_points[:,2]=0.1
    oldPoints[:,2]=0.2
    thres=1
    points_concat=torch.cat((current_points,oldPoints),dim=0) 
    nmsPoints=points_concat.T
    newpoints = torch.cat((nmsPoints[0:1, :] - thres, nmsPoints[1:2, :] - thres, nmsPoints[0:1, :] + thres,
                           nmsPoints[1:2, :] + thres, nmsPoints[2:3, :]), 0).T
    res = torchvision.ops.nms(newpoints[:, 0:4], newpoints[:, 4], 0.01)
    points=nmsPoints[:,res].T
    return points


def GetBatchMultipleHeatmap(confidenceMap,threshold,NMSthes=1,mode='batchLevel'):

    mask=confidenceMap>threshold
    prob =confidenceMap[mask]
    pred=torch.nonzero(mask)
    points = pred[:, 2:4]
    points=points.flip(1)
    if mode=='clustersLevel':
        idx=100*pred[:,0]+pred[:,1]
    elif mode=='batchLevel':
        idx =pred[:, 0]
    nmsPoints=torch.cat((points.float(),prob.unsqueeze(1)),1).T
    thres = math.ceil(NMSthes / 2)
    newpoints = torch.cat((nmsPoints[0:1, :] - thres, nmsPoints[1:2, :] - thres, nmsPoints[0:1, :] + thres,
                           nmsPoints[1:2, :] + thres, nmsPoints[2:3, :]), 0).T


    res = torchvision.ops.boxes.batched_nms(newpoints[:, 0:4], newpoints[:, 4],idx, 0.01)
    p=torch.cat((pred[res,:1].float(),nmsPoints[:,res].T,pred[res,1:2].float()),dim=1)
    value, indices = p[:,0].sort()
    p=p[indices]
    return p


def GetDescriptors(descriptor_volume, points, W, H):
    D = descriptor_volume.shape[0]
    if points.shape[0] == 0:
        descriptors = torch.zeros((0, D))
    else:
        coarse_desc = descriptor_volume.unsqueeze(0)
        samp_pts = points.clone().T
        samp_pts[0, :] = (samp_pts[0, :] / (float(W) / 2.)) - 1.
        samp_pts[1, :] = (samp_pts[1, :] / (float(H) / 2.)) - 1.
        samp_pts = samp_pts.transpose(0, 1).contiguous()
        samp_pts = samp_pts.view(1, 1, -1, 2)
        samp_pts = samp_pts.float()
        densedesc = torch.nn.functional.grid_sample(coarse_desc, samp_pts)
        densedesc = densedesc.view(D, -1)
        densedesc /= torch.norm(densedesc, dim=0).unsqueeze(0)
        descriptors = densedesc.T
    return descriptors


def GetPointsFromHeatmaps(heatmapOutput):
    # get max for each batch sample
    keypoints = torch.zeros(heatmapOutput.size(0), 4)

    val, idx = torch.max(heatmapOutput.view(heatmapOutput.shape[0], -1), 1)
    keypoints[:, 2] = val
    keypoints[:, :2] = idx.view(idx.size(0), 1).repeat(1, 1, 2).float()
    keypoints[..., 0] = (keypoints[..., 0] - 1) % heatmapOutput.size(2) + 1
    keypoints[..., 1] = keypoints[..., 1].add_(-1).div_(heatmapOutput.size(1)).floor()
    keypoints[:, 3] = torch.arange(heatmapOutput.size(0))

    keypoints[:, :2] = 4 * keypoints[:, :2]
    return keypoints


def fastDrawGaussian(img,pt,size):
    if (size == 3):
        g = gaussian3
    elif (size == 1):
        g = gaussian1
    s = 1
    ul = torch.tensor([[math.floor(pt[0] - s)], [math.floor(pt[1] -s)]])
    br = torch.tensor([[math.floor(pt[0] + s)], [math.floor(pt[1] +s)]])
    if (ul[0] > img.shape[1] or ul[1] > img.shape[0] or br[0] < 1 or br[1] < 1):
        return img

    g_x = torch.tensor([[max(1, -ul[0])], [min(br[0], img.shape[1]) - max(1, ul[0]) + max(1, -ul[0])]])
    g_y = torch.tensor([[max(1, -ul[1])], [min(br[1], img.shape[0]) - max(1, ul[1]) + max(1, -ul[1])]])
    img_x = torch.tensor([[max(1, ul[0])], [min(br[0], img.shape[1])]])
    img_y = torch.tensor([[max(1, ul[1])], [min(br[1], img.shape[0])]])

    assert (g_x[0] > 0 and g_y[0] > 0)
    img[int(img_y[0])-1:int(img_y[1]), int(img_x[0])-1:int(img_x[1])] += g[int(g_y[0])-1:int(g_y[1]), int(g_x[0])-1:int(g_x[1])]
    return img


def gaussian(size=3, sigma=0.25, amplitude=1, normalize=False, width=None,
             height=None, sigma_horz=None, sigma_vert=None, mean_horz=0.5, mean_vert=0.5):
        # handle some defaults
        if width is None:
            width = size
        if height is None:
            height = size
        if sigma_horz is None:
            sigma_horz = sigma
        if sigma_vert is None:
            sigma_vert = sigma
        center_x = mean_horz * width + 0.5
        center_y = mean_vert * height + 0.5
        gauss = np.empty((height, width), dtype=np.float32)
        # generate kernel
        for i in range(height):
            for j in range(width):
                gauss[i][j] = amplitude * math.exp(-(math.pow((j + 1 - center_x) / (sigma_horz * width), 2) / 2.0 + math.pow(
                    (i + 1 - center_y) / (sigma_vert * height), 2) / 2.0))
        if normalize:
            gauss = gauss / np.sum(gauss)
        return gauss


gaussian3=torch.tensor([[0.16901332, 0.41111228, 0.16901332],
       [0.41111228, 1.        , 0.41111228],
       [0.16901332, 0.41111228, 0.16901332]])


gaussian1=torch.tensor([[0.0, 0.0, 0.0],
       [0.0, 1.  , 0.0],
       [0.0, 0.0, 0.0]])



colorlist = ["#00ffff",
             "#ed1600",
             "#663be7",
             "#5abd3e",
             "#77c300",
             "#c647d7",
             "#521ac2",
             "#005e23",
             "#ff0092",
             "#cef39f",
             "#0046db",
             "#ff6100",
             "#c65cff",
             "#ffb200",
             "#ff93ff",
             "#5eb79c",
             "#ff0061",
             "#479197",
             "#f60034",
             "#436090",
             "#da906c",
             "#95959a",
             "#bf3c24",
             "#d752a3",
             "#3e542a",
             "#792d6a",
             "#9ba361",
             "#6c2928",
             "#3e542a",
             "#792d6a",
             "#9ba361",
             "#6c2928",
             "#da97b9",
             "#45051a"]
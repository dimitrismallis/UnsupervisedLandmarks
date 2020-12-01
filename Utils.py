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


def GetPathsForClusterVisualisation(experiment_name,log_path):
    CheckPointDirectory = log_path+'CheckPoints/' +experiment_name + '/*'
    listoffiles=glob.glob(CheckPointDirectory)
    path_to_keypoints=max([f for f in listoffiles if '.pickle' in f and 'UpdatedKeypoints' in f or 'Super' in f], key=os.path.getctime)
    LogText('Keypoints loaded from :'+(str(path_to_keypoints)),experiment_name,log_path)
    return path_to_keypoints


def GetPathsTrainSecondStep(experiment_name,log_path):
    CheckPointDirectory = log_path+'CheckPoints/' +experiment_name + '/*'
    listoffiles=glob.glob(CheckPointDirectory)

    path_to_checkpoint=max([f for f in listoffiles if '.pth' in f and 'FirstStep' in f ], key=os.path.getctime)
    path_to_keypoints=max([f for f in listoffiles if '.pickle' in f and 'UpdatedKeypoints' in f], key=os.path.getctime)
    LogText('Keypoints loaded from :'+(str(path_to_keypoints)),experiment_name,log_path)
    LogText('Checkpoint loaded from :'+(str(path_to_checkpoint)),experiment_name,log_path)
    return path_to_checkpoint,path_to_keypoints

def GetPathsResumeFirstStep(experiment_name,log_path):
    CheckPointDirectory = log_path+'CheckPoints/' +experiment_name + '/*'
    listoffiles=glob.glob(CheckPointDirectory)

    path_to_keypoints=max([f for f in listoffiles if '.pickle' in f and ('SuperPoint' in f or 'Updated' in f) ] , key=os.path.getctime)

    sortedlistoffiles=sorted(listoffiles,key=os.path.getctime)

    indexofcheckpoint=sortedlistoffiles.index(path_to_keypoints)
    sortedlistoffiles=sortedlistoffiles[:indexofcheckpoint]

    try:
        path_to_checkpoint=max([f for f in sortedlistoffiles if '.pth' in f and 'FirstStep' in f ], key=os.path.getctime)
    except:
        path_to_checkpoint=None
        LogText('Checkpoint was not found',experiment_name,log_path)

    LogText('Keypoints loaded from :'+(str(path_to_keypoints)),experiment_name,log_path)
    LogText('Checkpoint loaded from :'+(str(path_to_checkpoint)),experiment_name,log_path)
    return path_to_checkpoint,path_to_keypoints

def GetPathsResumeSecondStep(experiment_name,log_path):
    CheckPointDirectory = log_path+'CheckPoints/' +experiment_name + '/*'
    listoffiles=glob.glob(CheckPointDirectory)

    sortedlistoffiles=sorted(listoffiles,key=os.path.getctime)

    path_to_checkpoint=max([f for f in listoffiles if '.pth' in f and 'SecondStep' in f], key=os.path.getctime)

    indexofcheckpoint=sortedlistoffiles.index(path_to_checkpoint)
    sortedlistoffiles=sortedlistoffiles[:indexofcheckpoint]

    path_to_keypoints=max([f for f in sortedlistoffiles if '.pickle' in f and ('Merged' in f or 'Updated' in f) ] , key=os.path.getctime)
    LogText('Keypoints loaded from :'+(str(path_to_keypoints)),experiment_name,log_path)
    LogText('Checkpoint loaded from :'+(str(path_to_checkpoint)),experiment_name,log_path)
    return path_to_checkpoint,path_to_keypoints

def GetPathsEval(experiment_name,log_path):
    CheckPointDirectory = log_path+'CheckPoints/' +experiment_name + '/*'
    listoffiles=glob.glob(CheckPointDirectory)

    path_to_checkpoint=max([f for f in listoffiles if '.pth' in f and 'SecondStep' in f], key=os.path.getctime)
    LogText('Checkpoint loaded from :'+(str(path_to_checkpoint)),experiment_name,log_path)
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



colorlist = [
	"#ffdd41",
	"#0043db",
	"#62ef00",
	"#ff34ff",
	"#00ff5e",
	"#ef00de",
	"#00bd00",
	"#8f00c3",
	"#e5f700",
	"#a956ff",
	"#4bba00",
	"#ee00c2",
	"#4cbb0e",
	"#ff00c2",
	"#00ffa8",
	"#fe60f9",
	"#55b200",
	"#0052e5",
	"#ffe000",
	"#001e96",
	"#f1e215",
	"#336dff",
	"#e9d800",
	"#6056e7",
	"#ffd910",
	"#0070ff",
	"#8cbb00",
	"#0041c2",
	"#61bf2c",
	"#2a007b",
	"#00b64b",
	"#8237c0",
	"#00c87c",
	"#750091",
	"#00ffd3",
	"#f50000",
	"#00ffff",
	"#ff003a",
	"#00ffff",
	"#d40000",
	"#00ffff",
	"#c70000",
	"#00ffff",
	"#cb0008",
	"#00ffff",
	"#ed4000",
	"#00ffff",
	"#ff005d",
	"#00e8c5",
	"#d20022",
	"#00ffff",
	"#b00000",
	"#00f2f6",
	"#d50031",
	"#00edd2",
	"#ce004e",
	"#00c47f",
	"#7f56e3",
	"#ffa900",
	"#0046c7",
	"#d0b325",
	"#001175",
	"#ff8600",
	"#0080fb",
	"#ca9700",
	"#0060d9",
	"#ac9a00",
	"#006de2",
	"#f2a533",
	"#0088ff",
	"#d74900",
	"#0066d8",
	"#618a00",
	"#d679ff",
	"#077200",
	"#ff88ff",
	"#008a2d",
	"#590077",
	"#00d19a",
	"#c8005b",
	"#00f0e1",
	"#ac000f",
	"#00f0ff",
	"#a10000",
	"#00ecff",
	"#c12d0c",
	"#00e9ff",
	"#d34f00",
	"#005fd0",
	"#6a8600",
	"#0060cd",
	"#c67900",
	"#0066d1",
	"#9b8700",
	"#210052",
	"#ffe585",
	"#1a0045",
	"#00b87b",
	"#d665cf",
	"#005600",
	"#ff97ff",
	"#005100",
	"#ff9dff",
	"#005100",
	"#e685e5",
	"#1a5800",
	"#ffa6ff",
	"#004900",
	"#ff6eb9",
	"#00540e",
	"#d49dff",
	"#004500",
	"#ffb1ff",
	"#004300",
	"#dd337b",
	"#00e3d0",
	"#940000",
	"#00e8ff",
	"#ae191d",
	"#00e3f8",
	"#860000",
	"#70faff",
	"#860000",
	"#69f6fa",
	"#790000",
	"#00d6ff",
	"#a21221",
	"#00d6e0",
	"#bf1b3d",
	"#00d5dd",
	"#780002",
	"#00d3ff",
	"#730000",
	"#00d2ff",
	"#ac4f00",
	"#00a6ff",
	"#f37e3b",
	"#0081e3",
	"#496000",
	"#533d9a",
	"#138b53",
	"#7c005e",
	"#00814b",
	"#60005b",
	"#005919",
	"#ffb7ff",
	"#004000",
	"#ffb9ff",
	"#003800",
	"#cca8ff",
	"#003200",
	"#ff7bab",
	"#003200",
	"#ff7ca1",
	"#002f00",
	"#ff84a8",
	"#002900",
	"#cebbff",
	"#324d00",
	"#0094f1",
	"#765400",
	"#00aaff",
	"#693600",
	"#00adff",
	"#6b4400",
	"#0094eb",
	"#595900",
	"#006bc4",
	"#dbb670",
	"#001f6a",
	"#ffc49b",
	"#000432",
	"#eef6e3",
	"#230028",
	"#bef9ff",
	"#a90041",
	"#00bfc1",
	"#85002e",
	"#63dbe9",
	"#750024",
	"#00c7ff",
	"#480900",
	"#21c6ff",
	"#4c1400",
	"#00bdff",
	"#655400",
	"#0096e6",
	"#ffa078",
	"#001d57",
	"#ffac96",
	"#382b7f",
	"#374900",
	"#0091e2",
	"#3f4900",
	"#00bdff",
	"#360000",
	"#6adcff",
	"#350000",
	"#61d9ff",
	"#6c002f",
	"#00c4f8",
	"#813529",
	"#7cdbfd",
	"#1e0a00",
	"#ede4ff",
	"#001400",
	"#ece4ff",
	"#001b00",
	"#ffd9ef",
	"#002900",
	"#e1e1ff",
	"#002500",
	"#aeadee",
	"#002400",
	"#eaa6a5",
	"#002200",
	"#b0709f",
	"#004822",
	"#003a7d",
	"#a4965a",
	"#004b86",
	"#8c7d42",
	"#005b99",
	"#483000",
	"#008cc3",
	"#9a633e",
	"#005083",
	"#a68c61",
	"#001e3d",
	"#82764a",
	"#003054",
	"#1c3100",
	"#4d2c57",
	"#009b9e",
	"#2b1500",
	"#00929e",
	"#441c30",
	"#127c72",
	"#261700",
	"#005b88",
	"#2a2800",
	"#00456f",
	"#003c19",
	"#7f6788",
	"#062300",
	"#505880",
	"#004c29",
	"#00496e",
	"#2b2500",
	"#628687",
	"#020d14",
	"#005e6b",
	"#271b00",
	"#00546d",
	"#1d1617",
	"#264133",
	"#252737",
	"#002f24",
	"#002f3d",
	"#001919"
]
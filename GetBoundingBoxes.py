
import yaml
import glob
import cv2
import face_alignment
import numpy as np

with open('paths/LS3D.yaml') as file:
    paths = yaml.load(file, Loader=yaml.FullLoader)
datapath = paths['datapath']
files = glob.glob(datapath + '/**/*.jpg', recursive=True)+glob.glob(datapath + '/**/*.png', recursive=True)

face_detector='sfd'
face_detector_module=__import__('face_alignment.detection.'+face_detector,globals(),locals(),[face_detector],0)
face_detector=face_detector_module.FaceDetector(device='cuda',verbose=False)
import random
random.shuffle(files)

for imagefile in files:
    image = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)
    boundingbox=face_detector.detect_from_image(image[...,::-1].copy())[0]


    import matplotlib
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1)
    ax.set_axis_off()
    ax.imshow(image)
    # ax.scatter(keypoints[:, 0].cpu().detach().numpy(), keypoints[:, 1].cpu().detach().numpy())
    ax.scatter(np.array(boundingbox)[[0,2]], np.array(boundingbox)[[1,3]])
    plt.show()
    fig.savefig(f'/home/SERILOCAL/d.mallis/Projects/UnsupervisedLandmarks/foo.jpg')
    import time
    time.sleep(0.5)

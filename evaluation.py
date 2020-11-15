import numpy as np
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import random
from sklearn.linear_model import LinearRegression
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer
import Utils
import copy
import math
from Utils import LogText
from scipy import linalg
import numpy as np
import matplotlib.pyplot as plt
from matrix_completion import svt_solve, calc_unobserved_rmse
import pandas as pd


class Evaluator():
    def __init__(self, database_name,experiment_name,log_path):
        self.database_name=database_name
        self.experiment_name=experiment_name
        self.log_path=log_path

    
    def Evaluate(self,keypoints):
        if(self.database_name in ['CelebA']):
            N=300
            landmarksfornormalise=[36,45]

        forward_per_landmark_cumulative,backward_per_landmark_cumulative=self.evaluate_backward_forward(keypoints,landmarksfornormalise,N)
        
        if(self.database_name == 'CelebA'):
            titlebac=r"$\bf{MAFL}$, $\it{Backward}$"
            titlefor=r"$\bf{MAFL}$, $\it{Forward}$"

        import matplotlib.pyplot as plt
        import matplotlib.ticker as mtick
        fig = plt.figure(figsize=(6,6))
        ax = fig.gca()
        ax.set_facecolor('#F8F8F8')
        plt.title(titlebac, fontsize=24)
        plt.xlim(1, len(backward_per_landmark_cumulative)-1)
        ax.tick_params(labelsize=14)
        plt.grid()
        plt.plot(np.arange(1, len(backward_per_landmark_cumulative) + 1), 100 * backward_per_landmark_cumulative,  c='red', linewidth=10)
        plt.ylabel('NME (%)', fontsize=20, fontstyle='italic')
        plt.xlabel('# unsupervised object landmarks', fontsize=20, fontstyle='italic')
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        plt.tight_layout()
        plt.show()
        fig.savefig(self.log_path + 'Logs/' + self.experiment_name +'/'+self.database_name+ f'BackwardEpoch.jpg')


        import matplotlib.pyplot as plt
        import matplotlib.ticker as mtick
        fig = plt.figure(figsize=(6,6))
        ax = fig.gca()
        ax.set_facecolor('#F8F8F8')
        plt.title(titlefor, fontsize=24)
        ax.tick_params(labelsize=14)
        plt.grid()         
        plt.plot(np.arange(1, len(forward_per_landmark_cumulative) + 1), 100 * forward_per_landmark_cumulative, c='red', linewidth=10)          
        plt.ylabel('NME (%)', fontsize=20, fontstyle='italic')
        plt.xlabel('# of groundtruth landmarks', fontsize=20, fontstyle='italic')
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        plt.tight_layout()
        plt.show()
        fig.savefig(self.log_path + 'Logs/' + self.experiment_name +'/'+self.database_name+ f'ForwardEpoch.jpg')
        return



    def evaluate_backward_forward(self,points,landmarksfornormalise,N):
        keypoints=copy.deepcopy(points)
        Samples=[f for f in keypoints.keys() ]

        number_of_detected_keypoints = len(keypoints[Samples[0]]['prediction'])
        number_of_groundtruth_points = len(keypoints[Samples[0]]['groundtruth'])

        keypoints_array = np.zeros((len(Samples), 2 * number_of_detected_keypoints))
        groundtruth_array = np.zeros((len(Samples), 2 * number_of_groundtruth_points))
        is_test_sample=np.zeros(len(Samples))

        for i in range(len(Samples)):
            sample_points=keypoints[Samples[i]]['prediction']
            keypoints_array[i]=sample_points.reshape(-1)
            is_test_sample[i]=keypoints[Samples[i]]['is_it_test_sample']
            sample_gt = keypoints[Samples[i]]['groundtruth']
            groundtruth_array[i]=sample_gt.reshape(-1)

        #clusters that are detected in less than 20% of images are not considered in the evaluation
        keypoints_array=keypoints_array[:,np.sum(np.isnan(keypoints_array),axis=0)<0.2*len(keypoints_array)]

        backward_per_landmark = Backward(keypoints_array,groundtruth_array,groundtruth_array, is_test_sample, N,landmarksfornormalise)
        forward_per_landmark= Forward_matrix(keypoints_array,groundtruth_array,groundtruth_array, is_test_sample, N,landmarksfornormalise)

        backward_per_landmark.sort()
        backward_per_landmark_cumulative = np.cumsum(backward_per_landmark)
        backward_per_landmark_cumulative = backward_per_landmark_cumulative / np.arange(1, len(backward_per_landmark) + 1)

        forward_per_landmark.sort()
        forward_per_landmark_cumulative = np.cumsum(forward_per_landmark)
        forward_per_landmark_cumulative = forward_per_landmark_cumulative / np.arange(1, len(forward_per_landmark) + 1)

        return forward_per_landmark_cumulative,backward_per_landmark_cumulative




def Backward(keypoints_array,
             groundtruth_array,
             groundtruth_array_for_normalisation,
             is_test_sample,
             N,
             landmarksfornormalise=None):

    keypoints_array=keypoints_array.copy()
    groundtruth_array=groundtruth_array.copy()
    groundtruth_array_for_normalisation=groundtruth_array_for_normalisation.copy()

    keypoints_array = keypoints_array.reshape(keypoints_array.shape[0], -1, 2)

    groundtruth_array_for_normalisation = groundtruth_array_for_normalisation[is_test_sample==1]


    backward_per_landmark = np.zeros(keypoints_array.shape[1])

    train_keypoints_array=keypoints_array[is_test_sample==0][:N]
    test_keypoints_array = keypoints_array[is_test_sample==1]

    test_groundtruth = groundtruth_array[is_test_sample==1]
    train_groundtruth = groundtruth_array[is_test_sample == 0][:N]

    number_of_landmarks=len(backward_per_landmark)
    number_of_confident_instances_per_landmarks=np.zeros(len(backward_per_landmark))
    for j in range(number_of_landmarks):

        train_keypoints_array_forlandmark=train_keypoints_array[:,j]

        landmarknotnan=(~np.isnan(train_keypoints_array_forlandmark))[:, 0]

        train_keypoints_array_forlandmark=train_keypoints_array_forlandmark[landmarknotnan]
        groundtruth_array_forlanamrk=train_groundtruth[landmarknotnan]

        R_backward, X0_backward, Y0_backward=train_regressor(groundtruth_array_forlanamrk,train_keypoints_array_forlandmark,0.01,256,'type2')
        landmarkbackward=0
        count=0
        for i in range(len(test_keypoints_array)):
            point=test_keypoints_array[i,j]
            point_gt=test_groundtruth[i]
            gt_fornormal=groundtruth_array_for_normalisation[i].reshape(-1,2)
            if(np.isnan(point)[0]==False):
                y_predict=fit_regressor(R_backward,point_gt,X0_backward, Y0_backward,256,'type2')
                normalisedistance=GetnormaliseDistance(gt_fornormal,landmarksfornormalise)
                
                distance = np.sqrt(np.sum((point -y_predict) ** 2, axis=-1))/normalisedistance
                landmarkbackward+=distance
                count+=1
        
        if(count==0):
            landmarkbackward=1
        else:
            landmarkbackward=landmarkbackward/count
        backward_per_landmark[j]=landmarkbackward
    return backward_per_landmark




def Forward_matrix(keypoints_array,
             groundtruth_array,
             groundtruth_array_for_normalisation,
             is_test_sample,
             N,
             landmarksfornormalise=None,
             number_of_different_landmarks=3):

    keypoints_array=keypoints_array.copy()

    groundtruth_array=groundtruth_array.copy()
    groundtruth_array_for_normalisation=groundtruth_array_for_normalisation.copy()

    keypoints_array = keypoints_array.reshape(keypoints_array.shape[0], -1, 2)

    groundtruth_array_for_normalisation = groundtruth_array_for_normalisation[is_test_sample==1]

    forward_per_landmark = np.zeros(int(groundtruth_array.shape[1]/2))


    train_keypoints_array=keypoints_array[is_test_sample==0]

    test_keypoints_array = keypoints_array[is_test_sample==1]

    test_groundtruth = groundtruth_array[is_test_sample==1]
    train_groundtruth = groundtruth_array[is_test_sample == 0]


    number_of_test_samples=len(test_keypoints_array)

    nl = 2*keypoints_array.shape[1] 

    Xtr_new = train_keypoints_array
    Xtr_new=Xtr_new.reshape(Xtr_new.shape[0],-1)
    Xtest_new = test_keypoints_array.reshape(test_keypoints_array.shape[0],keypoints_array.shape[1],2)

    DF = pd.DataFrame(Xtr_new)
    col_means = DF.apply(np.mean, 0)
    Xc_tr_mean = DF.fillna(value=col_means).to_numpy()/256.0
    Xc_tr = Xc_tr_mean.copy()
    mask = np.ones_like(Xtr_new.reshape(len(Xtr_new),nl))
    mask[np.where(np.isnan(Xtr_new.reshape(len(Xtr_new),nl)))] = 0

    R_hat = svt_solve(Xc_tr, np.round(mask))
    Xc_tr = 256.0 * R_hat

    Xc_tr[np.where(mask==1)] = Xtr_new.reshape(len(Xtr_new),nl)[np.where(mask==1)]

    DF = pd.DataFrame(Xtest_new.reshape(Xtest_new.shape[0],nl))
    Xc_test = DF.fillna(value=col_means).to_numpy()
    Ytest=test_groundtruth
    err_fwd_fs = np.zeros((100,Xc_test.shape[0],Ytest.shape[1]//2))
    err_fwd_io = np.zeros((100,Xc_test.shape[0],Ytest.shape[1]//2))

    for j in range(0,100):
        reg_factor = 0.01
        ty = 'type2'
        centre = 256.0
        imgs = np.random.permutation(1000)[:N]
        Ytr_aux = train_groundtruth[imgs,:]
        Xc_tr_aux = Xc_tr[imgs,:]
        R, X0, Y0 = train_regressor(Xc_tr_aux, Ytr_aux, reg_factor, centre, ty)
        
        for i in range(0,test_keypoints_array.shape[0]):
            x = Xc_test[i,:]
            y = test_groundtruth[i,:]
            x = fit_regressor(R,x,X0,Y0,centre,ty)
            gt = y.reshape(-1,2)

            iod=GetnormaliseDistance(gt,landmarksfornormalise)
            y = y.reshape(-1,2)

            err_fwd_io[j,i,:] =np.sqrt(np.sum((x-y)**2,1))/iod

    err_fwd_io = np.mean(np.mean(err_fwd_io,axis=0),axis=0)
    return err_fwd_io


def GetnormaliseDistance(gt_fornormal,landmarksfornormalise):
    if(landmarksfornormalise is None):
        #use size of the bounding box
        h=np.max(gt_fornormal[:,1])-np.min(gt_fornormal[:,1])
        w=np.max(gt_fornormal[:,0])-np.min(gt_fornormal[:,0])
        normdistance=math.sqrt(h*w)
    else:
        eyes = gt_fornormal[ landmarksfornormalise, :]
        normdistance = np.sqrt(np.sum((eyes[ 0, :] - eyes[ 1, :]) ** 2, axis=-1))
    return normdistance


def train_regressor(X,Y,l,center=128.0,option=None):
    if option == 'type0':
        C = X.transpose() @ X
        R = ( Y.transpose() @ X ) @ linalg.inv( C + l*(C.max()+1e-12)*np.eye(X.shape[1]))
        X0 = 1.0
        Y0 = 1.0
    elif option == 'type1':
        Xtmp = X/center - 0.5
        C = Xtmp.transpose() @ Xtmp
        Ytmp = Y/center - 0.5
        R = ( Ytmp.transpose() @ Xtmp ) @ linalg.inv( C + l*(C.max()+1e-12)*np.eye(Xtmp.shape[1]))
        X0 = 1.0
        Y0 = 1.0
    elif option == 'type2':
        Xtmp = X/center - 0.5
        X0 = Xtmp.mean(axis=0, keepdims=True)
        Xtmp = Xtmp - np.ones((Xtmp.shape[0],1)) @ X0.reshape(1,-1)
        C = Xtmp.transpose() @ Xtmp
        Ytmp = Y/center - 0.5
        Y0 = Ytmp.mean(axis=0, keepdims=True)
        Ytmp = Ytmp - np.ones((Ytmp.shape[0],1)) @ Y0.reshape(1,-1)
        R = ( Ytmp.transpose() @ Xtmp ) @ linalg.inv( C + l*(C.max()+1e-12)*np.eye(Xtmp.shape[1])) 
    elif option == 'type3':
        Xtmp = X
        X0 = Xtmp.mean(axis=0, keepdims=True)
        Xtmp = Xtmp - np.ones((Xtmp.shape[0],1)) @ X0.reshape(1,-1)
        C = Xtmp.transpose() @ Xtmp
        Ytmp = Y
        Y0 = Ytmp.mean(axis=0, keepdims=True)
        Ytmp = Ytmp - np.ones((Ytmp.shape[0],1)) @ Y0.reshape(1,-1)
        R = ( Ytmp.transpose() @ Xtmp ) @ linalg.inv( C + l*(C.max()+1e-12)*np.eye(Xtmp.shape[1]))
    return R, X0, Y0


def fit_regressor(R,x,X0,Y0,center=128.0,option=None):
    if option == 'type0':
        x = (R @ x).reshape(-1,2)
    elif option == 'type1':
        x = (R @ (x/center - 0.5).transpose()).reshape(-1,2)
        x = (x + 0.5)*center
    elif option == 'type2':
        x = (R @ (x/center - 0.5 - X0).transpose()).reshape(-1,2) + Y0.reshape(-1,2)
        x = (x + 0.5)*center
    elif option == 'type3':
        x = (R @ (x - X0).transpose()).reshape(-1,2) + Y0.reshape(-1,2)
    return x



import faiss,math
import numpy as np
import time

class Kmeans(object):
    def __init__(self, k,centroids=None,dimension=256):
        self.k = k
        self.centroids=centroids
        self.clus = faiss.Clustering(dimension, self.k)

        self.clus.seed = 0
        self.clus.nredo = 1
        self.clus.niter = 100


    def cluster(self, data, verbose=False,featuredimension=256):
        end = time.time()

        xb=data

        # cluster the data
        I, index,centroids= self.run_kmeans(xb, verbose)
        self.index=index
        self.images_lists = [[] for i in range(self.k)]
        for i in range(len(data)):
            self.images_lists[I[i]].append(i)

        if verbose:
            print('k-means time: {0:.0f} s'.format(time.time() - end))

        return self.images_lists,faiss.vector_to_array(centroids).reshape(self.k ,featuredimension)

    def run_kmeans(self,x, verbose=False):
        n_data, d = x.shape
        res = faiss.StandardGpuResources()
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.useFloat16 = False
        flat_config.device = 0
        index = faiss.IndexFlatL2(d)
        gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index)
        # perform the training
        self.clus.train(x, gpu_index_flat)

        centroids = self.clus.centroids
        _, I = gpu_index_flat.search(x, 1)
        if verbose:
            print('k-means loss evolution: {0}'.format(losses))

        return [int(n[0]) for n in I], gpu_index_flat, centroids


def preprocess_features(npdata):

    _, ndim = npdata.shape
    npdata =  npdata.astype('float32')

    # L2 normalization
    row_sums = np.linalg.norm(npdata, axis=1)
    npdata = npdata / row_sums[:, np.newaxis]

    return npdata
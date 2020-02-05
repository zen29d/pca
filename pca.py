
import numpy as np

class PCA:
    data_ = None
    cov_mat_ = None
    pca_component = None
    variance_ratio = None
    trans_data = None

    def __init__(self, n_comp=None):
        self.n_comp = n_comp

    def cov_mat(data, ddof=1):
        '''return covariance matrix of data
           default ddof = 1'''
        import numpy as np
        data = data.T
        if ddof:
            size = data.shape[0]-1
        else:
            size = data.shape[0]
        mean = np.mean(data, axis=0)
        cov_mat = (data-mean).T.dot(data-mean)/size
        return cov_mat

    def pca_comp(data):
        '''return pca component of square matrix'''
        eig_vals, eig_vecs = np.linalg.eig(data)
        eig_vals = eig_vals.reshape([1,len(eig_vals)])
        mat = np.concatenate((eig_vals,eig_vecs), axis=0).T
        mat = mat[mat[:,0].argsort()][::-1]
        component = mat[:,1:]
        # if n_comp is None:
        #     component = mat[:,1:]
        # else:
        #     component = mat[:,1:]
        return component
        
    #variance_ratio
    def var_ratio():
        '''variance ratio'''
        eig_vals, _ = np.linalg.eig(PCA.cov_mat_)
        var_ratio = [i/sum(eig_vals) for i in sorted(eig_vals, reverse=True)]
        return var_ratio

    #initialize the operation for pca
    @staticmethod
    def fit(data):
        PCA.data_ = data
        PCA.cov_mat_ = PCA.cov_mat(PCA.data_) 
        PCA.pca_component = PCA.pca_comp(PCA.cov_mat_)
        PCA.variance_ratio = PCA.var_ratio()

    #transform the data
    @staticmethod
    def transform(data):
        '''transform data with pca_component
            data is not normalized'''
        PCA.trans_data = np.dot(PCA.data_.T,PCA.pca_component)
        return PCA.trans_data

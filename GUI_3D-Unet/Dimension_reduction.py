import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA, SparsePCA, TruncatedSVD
def DRMethod(method, HSI, NC = 75):
    RHSI = np.reshape(HSI, (-1, HSI.shape[2]))
    if method == 'PCA': ## PCA
        pca = PCA(n_components = NC, whiten = True)
        RHSI = pca.fit_transform(RHSI)
        RHSI = np.reshape(RHSI, (HSI.shape[0], HSI.shape[1], NC))
    elif method == 'iPCA': ## Incremental PCA
        n_batches = 256
        inc_pca = IncrementalPCA(n_components = NC)
        for X_batch in np.array_split(RHSI, n_batches):
          inc_pca.partial_fit(X_batch)
        X_ipca = inc_pca.transform(RHSI)
        RHSI = np.reshape(X_ipca, (HSI.shape[0], HSI.shape[1], NC))
    elif method == 'KPCA': ## Kernel PCA
        kpca = KernelPCA(kernel = "rbf", n_components = NC, gamma = None,
                         fit_inverse_transform = True, random_state = 2019,
                         n_jobs=1)
        kpca.fit(RHSI)
        RHSI = kpca.transform(RHSI)
        RHSI = np.reshape(RHSI, (HSI.shape[0], HSI.shape[1], NC))
    elif method == 'SPCA': ## Sparse PCA
        sparsepca = SparsePCA(n_components = NC, alpha=0.0001, random_state=2019, n_jobs=-1)
        sparsepca.fit(RHSI)
        RHSI = sparsepca.transform(RHSI)
        RHSI = np.reshape(RHSI, (HSI.shape[0], HSI.shape[1], NC))
    elif method == 'SVD': ## Singular Value Decomposition
        SVD_ = TruncatedSVD(n_components = NC,algorithm = 'randomized',
                            random_state = 2019, n_iter=5)
        SVD_.fit(RHSI)
        RHSI = SVD_.transform(RHSI)
        RHSI = np.reshape(RHSI, (HSI.shape[0], HSI.shape[1], NC))
    return RHSI

def Perform_DR(method,NC,data):
    X_DR = []
    if method == "NoDR":
        X_DR = data
    else:
        for i in data:
            X_DR.append(DRMethod(method, i, NC))
        X_DR = np.asarray(X_DR)
    return X_DR
#x_train=np.load('x_train_3DUnet.npy')
#inputHSI= x_train[:,:,:,3:]
#Perform_DR('PCA',60,inputHSI)
#print(np.shape(inputHSI))

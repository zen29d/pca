import numpy as np
from pca import PCA



# a =[10,30,60,20,40] 
# b = [20,10,20,40,50]
#-------------------------------
a = [2.5, 0.5, 2.2, 1.9, 3.1, 2.3, 2, 1, 1.5, 1.1]
b = [2.4, 0.7, 2.9, 2.2, 3, 2.7, 1.6, 1.1, 1.6, 0.9]
c = [2.4, 0.9, 2.1, 2, 3, 2.7, 1.6, 1.1, 1.6, 1.9]



X = np.array([a,b,c])
model = PCA(n_comp=1)
model.fit(X)
print(model.pca_component)
print(model.transform(X))
print(model.variance_ratio)
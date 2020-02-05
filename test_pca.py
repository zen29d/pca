import numpy as np
from pca import PCA

a = [2.5, 0.5, 2.2, 1.9, 3.1, 2.3, 2, 1, 1.5, 1.1]
b = [2.4, 0.7, 2.9, 2.2, 3, 2.7, 1.6, 1.1, 1.6, 0.9]
c = [2.4, 0.9, 2.1, 2, 3, 2.7, 1.6, 1.1, 1.6, 1.9]

X = np.array([a,b,c])
model = PCA()
model.fit(X)
print(model.pca_component)
print(model.transform(X))
print(model.variance_ratio)
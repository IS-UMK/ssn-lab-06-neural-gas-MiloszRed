import numpy as np
from sklearn.base import BaseEstimator
import matplotlib.pyplot as plt
from utils import img_to_vectors, vectors_to_image
import imageio

def square_euclid(x, y):
    return np.sum((x - y)**2, axis=1)

def mse(img1, img2):
    return np.sum(square_euclid(img1, img2))/img1.size

class VectorQuantization(BaseEstimator):
    def __init__(self, n_prototypes=10, n_epochs=10, eta0=0.5, eta_min=0.01, lambda0=5.0, lambda_min=0.5):
        self.k = n_prototypes
        self.N = n_epochs
        self.eta0 = eta0
        self.eta_min = eta_min
        self.lambda0 = lambda0
        self.lambda_min = lambda_min
    
    def init_prototypes(self, X):
        self.prototypes = np.random.permutation(X)[:self.k].copy()
        return self

    def find_nearest_prototype(self, x):
        dists = square_euclid(x, self.prototypes)
        return np.argmin(dists)

    def fit(self, X):
        self.init_prototypes(X)
        T = self.N * len(X)
        t = 0
        self.errors = [ self.score(X) ]
        
        for epoch in range(self.N):
            for x in np.random.permutation(X):
                eta_t = self.eta0 * (self.eta_min / self.eta0) ** (t / T)
                lambda_t = self.lambda0 * (self.lambda_min / self.lambda0) ** (t / T)

                dists = square_euclid(x, self.prototypes)
                ranks = np.argsort(dists)

                for r, i in enumerate(ranks):
                    h = np.exp(-r / lambda_t)
                    self.prototypes[i] += eta_t * h * (x - self.prototypes[i])

                t += 1

            self.errors.append(self.score(X))
        return self

    def predict(self, X):
        
        return np.array([self.find_nearest_prototype(x) for x in X], dtype=np.int32)

    def score(self, X):
        error = []
        for x in X:
            dists = square_euclid(x, self.prototypes)
            m = np.argmin(dists)
            error.append(dists[m])
        return np.mean(error)


image = imageio.imread('dane/Lenna.png')

patch_size = (5, 5)
n_prototypes = 200
X = img_to_vectors(image, patch_size=patch_size).astype(np.float64)

vq = VectorQuantization(n_prototypes=n_prototypes, n_epochs=10, eta0=0.5, eta_min=0.01, lambda0=5.0, lambda_min=0.05)
vq.fit(X)
idxs = vq.predict(X)

quantized_vectors = vq.prototypes[idxs]
restored_image = vectors_to_image(quantized_vectors, image.shape, patch_size=patch_size)

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title("Oryginalny obraz")
plt.imshow(image)

plt.subplot(1,2,2)
plt.title(f'Po kompresji (gaz neuronowy). Ksiega kodow {n_prototypes} blad rekonstrukcji MSE={mse(image, restored_image)}')
plt.imshow(restored_image)
plt.show()

plt.plot(range(len(vq.errors)), vq.errors)
plt.show()

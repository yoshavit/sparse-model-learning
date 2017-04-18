import numpy as np
from scipy.spatial.distance import cdist

class LatentVisualizer:

    def __init__(self):
        self.images = None
        self.latents = None

    def add_images(self, images, latents):
        """
        images - list of images
        latents - list of latent arrays
        """
        if not self.images:
            self.images = list(images)
            self.latents = np.stack(latents, axis=0)
        else:
            self.images.extend(images)
            self.latents = np.concatenate((self.latents, np.stack(latents,
                                                                  axis=0)))

    def get_nearest_image(self, latent):
        distances = cdist(self.latents, latent, "euclidean")
        closest_ind = np.argmax(distances)
        return self.images[closest_ind]

    def save(self, filename):
        np.savez(filename, images=self.images, latents=self.latents)

    def load(self, filename):
        data = np.load(filename)
        self.add_images(data['images'], data['latents'])

import numpy as np
import os
from scipy.spatial.distance import cdist

file_suffix = "state_latents.npz"
class LatentVisualizer:

    def __init__(self):
        self.images = None
        self.latents = None

    def add_images(self, images, latents):
        """
        images - nx?x?x? array of images
        latents - nx? array of latents 
        """
        if not self.images:
            self.images = images
            self.latents = latents
        else:
            self.images = np.concatenate((self.images, images))
            self.latents = np.concatenate((self.latents, latents))

    def get_nearest_image(self, latent):
        if latent.ndim == 1:
            latent = np.expand_dims(latent, 0)
        distances = cdist(self.latents, latent, "euclidean")
        closest_ind = np.argmin(distances)
        return self.images[closest_ind]

    def save(self, filename):
        filename = os.path.join(filename, file_suffix)
        if os.path.exists(filename):
            os.remove(filename)
        np.savez(filename, images=self.images, latents=self.latents)

    def load(self, filename):
        filename = os.path.join(filename, file_suffix)
        data = np.load(filename)
        self.add_images(data['images'], data['latents'])

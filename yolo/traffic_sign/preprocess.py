from sklearn.utils import shuffle
from skimage import exposure
import warnings
import numpy as np
from traffic_sign.utils import print_progress
from traffic_sign.dataset import readTrafficSigns

num_classes = 43


import random


class AugmentedSignsBatchIterator(BatchIterator):
    """
    Iterates over dataset in batches.
    Allows images augmentation by randomly rotating, applying projection,
    adjusting gamma, blurring, adding noize and flipping horizontally.
    """

    def __init__(self, batch_size, shuffle=False, seed=42, p=0.5, intensity=0.5):
        """
        Initialises an instance with usual iterating settings, as well as data augmentation coverage
        and augmentation intensity.

        Parameters
        ----------
        batch_size:
                    Size of the iteration batch.
        shuffle   :
                    Flag indicating if we need to shuffle the data.
        seed      :
                    Random seed.
        p         :
                    Probability of augmenting a single example, should be in a range of [0, 1] .
                    Defines data augmentation coverage.
        intensity :
                    Augmentation intensity, should be in a [0, 1] range.

        Returns
        -------
        New batch iterator instance.
        """
        super(AugmentedSignsBatchIterator, self).__init__(batch_size, shuffle, seed)
        self.p = p
        self.intensity = intensity

    def transform(self, Xb, yb):
        """
        Applies a pipeline of randomised transformations for data augmentation.
        """
        Xb, yb = super(AugmentedSignsBatchIterator, self).transform(
            Xb if yb is None else Xb.copy(),
            yb
        )

        if yb is not None:
            batch_size = Xb.shape[0]
            image_size = Xb.shape[1]

            Xb = self.rotate(Xb, batch_size)
            Xb = self.apply_projection_transform(Xb, batch_size, image_size)

        return Xb, yb

    def rotate(self, Xb, batch_size):
        """
        Applies random rotation in a defined degrees range to a random subset of images.
        Range itself is subject to scaling depending on augmentation intensity.
        """
        for i in np.random.choice(batch_size, int(batch_size * self.p), replace=False):
            delta = 30. * self.intensity  # scale by self.intensity
            Xb[i] = rotate(Xb[i], random.uniform(-delta, delta), mode='edge')
        return Xb

    def apply_projection_transform(self, Xb, batch_size, image_size):
        """
        Applies projection transform to a random subset of images. Projection margins are randomised in a range
        depending on the size of the image. Range itself is subject to scaling depending on augmentation intensity.
        """
        d = image_size * 0.3 * self.intensity
        for i in np.random.choice(batch_size, int(batch_size * self.p), replace=False):
            tl_top = random.uniform(-d, d)  # Top left corner, top margin
            tl_left = random.uniform(-d, d)  # Top left corner, left margin
            bl_bottom = random.uniform(-d, d)  # Bottom left corner, bottom margin
            bl_left = random.uniform(-d, d)  # Bottom left corner, left margin
            tr_top = random.uniform(-d, d)  # Top right corner, top margin
            tr_right = random.uniform(-d, d)  # Top right corner, right margin
            br_bottom = random.uniform(-d, d)  # Bottom right corner, bottom margin
            br_right = random.uniform(-d, d)  # Bottom right corner, right margin

            transform = ProjectiveTransform()
            transform.estimate(np.array((
                (tl_left, tl_top),
                (bl_left, image_size - bl_bottom),
                (image_size - br_right, image_size - br_bottom),
                (image_size - tr_right, tr_top)
            )), np.array((
                (0, 0),
                (0, image_size),
                (image_size, image_size),
                (image_size, 0)
            )))
            Xb[i] = warp(Xb[i], transform, output_shape=(image_size, image_size), order=1, mode='edge')

        return Xb

X_train, y_train = readTrafficSigns('./traffic-signs-data/Final_Training/Images')
X_train = X_train / 255.

batch_iterator = AugmentedSignsBatchIterator(batch_size = 5, p = 1.0, intensity = 0.75)
for x_batch, y_batch in batch_iterator(X_train, y_train):
    for i in range(5):
        # plot two images:
        fig = figure(figsize=(3, 1))
        axis = fig.add_subplot(1, 2, 1, xticks=[], yticks=[])
        axis.imshow(X_train[i])
        axis = fig.add_subplot(1, 2, 2, xticks=[], yticks=[])
        axis.imshow(x_batch[i])
        plt.show()
    break
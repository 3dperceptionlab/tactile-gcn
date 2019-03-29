import torch
import numpy as np
from copy import deepcopy

class AddNoise(object):

    def __init__(self, ff_noise=287, mf_noise=197, th_noise=236):
        self.ff_noise = ff_noise
        self.mf_noise = mf_noise
        self.th_noise = th_noise

    def __call__(self, sample):
        ff_noise_ = np.random.randint(-self.ff_noise, high=self.ff_noise, size=sample.x[:, 0].shape)
        mf_noise_ = np.random.randint(-self.mf_noise, high=self.mf_noise, size=sample.x[:, 1].shape)
        th_noise_ = np.random.randint(-self.th_noise, high=self.th_noise, size=sample.x[:, 2].shape)
        noise_ = np.array([ff_noise_, mf_noise_, th_noise_]).T

        augmented_batch_ = deepcopy(sample)
        augmented_batch_.x = augmented_batch_.x + torch.from_numpy(noise_).type(torch.FloatTensor)

        return augmented_batch_

    def __repr__(self):
        return "{}".format(self.__class__.__name__)
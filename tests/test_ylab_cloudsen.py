#!/usr/bin/env python

"""Tests for `ylab_cloudsen` package."""


import unittest

import numpy as np

from ylab_cloudsen import crop_functions


class CroppingAndReconstruction(unittest.TestCase):
    """Tests for cropping, reconstruction functions"""

    def gen_data(self, dim_x, patch_size, overlap):
        img = np.random.random(dim_x)
        img_out = np.zeros(dim_x)
        inds = crop_functions.prepare_indices_one_dim(dim_x, patch_size, overlap)
        for i in range(inds.shape[0]):
            ind_row = inds[i, :]
            patch = img[ind_row[0] : ind_row[0] + patch_size]
            img_out[ind_row[1] : ind_row[1] + ind_row[3]] = patch[ind_row[2] : ind_row[2] + ind_row[3]]
        return img, img_out

    def test_1d_reconstruction(self):
        res = self.gen_data(130, 25, 4)
        self.assertTrue((res[0] == res[1]).all())
        res = self.gen_data(131, 25, 4)
        self.assertTrue((res[0] == res[1]).all())
        res = self.gen_data(129, 35, 4)
        self.assertTrue((res[0] == res[1]).all())
        res = self.gen_data(129, 35, 6)
        self.assertTrue((res[0] == res[1]).all())
        res = self.gen_data(10890, 512, 32)
        self.assertTrue((res[0] == res[1]).all())

    def gen_data_2d(self, dims, patch_size, overlap):
        img = np.random.random(dims[0] * dims[1])
        img = np.reshape(img, dims)
        img_out = np.zeros(dims)
        inds = crop_functions.prepare_indices_2d(dims, patch_size, overlap)
        for i in range(inds.shape[0]):
            ind = inds[i, :]
            patch = img[ind[0] : ind[0] + patch_size[0], ind[1] : ind[1] + patch_size[1]]
            img_out[ind[2] : ind[2] + ind[6], ind[3] : ind[3] + ind[7]] = patch[
                ind[4] : ind[4] + ind[6], ind[5] : ind[5] + ind[7]
            ]
        return img, img_out

    def test_2d_reconstruction(self):
        res = self.gen_data_2d((130, 130), (25, 25), (4, 4))
        self.assertTrue((res[0] == res[1]).all())
        res = self.gen_data_2d((129, 131), (25, 25), (4, 4))
        self.assertTrue((res[0] == res[1]).all())
        res = self.gen_data_2d((10890, 10890), (512, 512), (32, 32))
        self.assertTrue((res[0] == res[1]).all())

    def prep_images(self, dims=(130, 130), patch_s=25, overlap=4):
        img = np.random.random(dims[0] * dims[1])
        img = np.reshape(img, dims)
        img = np.vstack([img, img])
        img = np.reshape(img, (2, *dims))
        img_out = np.zeros(img.shape)
        return img, img_out

    def test_insert_patches(self):
        dims = (130, 130)
        patch_s = 25
        overlap = 4
        img, img_out = self.prep_images(dims, patch_s, overlap)
        inds = crop_functions.prepare_indices_2d(dims, patch_s, overlap)
        patches = crop_functions.crop_patches(img, inds)
        # Actual reconstruction
        crop_functions.insert_patches(img_out, patches, inds)
        self.assertTrue((img == img_out).all())

    def test_batched_insert_patches(self):
        dims = (130, 130)
        patch_s = 25
        overlap = 4
        img, img_out = self.prep_images(dims, patch_s, overlap)
        inds = crop_functions.prepare_indices_2d(dims, patch_s, overlap)
        # Prepare batches
        inds_batches = crop_functions.indices_to_batches(inds, 7)
        for batch in inds_batches:
            patches = crop_functions.crop_patches(img, batch)
            # Actual reconstruction
            crop_functions.insert_patches(img_out, patches, batch)
        self.assertTrue((img == img_out).all())

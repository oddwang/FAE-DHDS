# -------------------------------------------------
# IMPORTS
# -------------------------------------------------

from data_utils import *
import numpy as np

# -------------------------------------------------
# SHIFT APPLICATOR
# -------------------------------------------------


def apply_shift(X_te_orig, y_te_orig, shift, orig_dims, datset):
	X_te_1 = X_te_orig.copy()
	y_te_1 = y_te_orig.copy()

	if shift == 'rand':
		print('Randomized')
		X_te_1 = X_te_orig.copy()
		y_te_1 = y_te_orig.copy()
	elif shift == 'large_gn_shift_1.0':
		print('Large GN shift 1.0')
		normalization = 255.0
		X_te_1, _ = gaussian_noise_subset(X_te_orig, 100.0, normalization=normalization, delta_total=1.0)
		y_te_1 = y_te_orig.copy()
	elif shift == 'medium_gn_shift_1.0':
		print('Medium GN Shift 1.0')
		normalization = 255.0
		X_te_1, _ = gaussian_noise_subset(X_te_orig, 10.0, normalization=normalization, delta_total=1.0)
		y_te_1 = y_te_orig.copy()
	elif shift == 'small_gn_shift_1.0':
		print('Small GN Shift 1.0')
		normalization = 255.0
		# normalization = np.std(X_te_orig)
		X_te_1, _ = gaussian_noise_subset(X_te_orig, 1.0, normalization=normalization, delta_total=1.0)
		y_te_1 = y_te_orig.copy()
	elif shift == 'tiny_gn_shift_1.0':
		print('Tiny GN Shift 1.0')
		noise_amt = 0.5
		normalization = np.std(X_te_orig)
		noise = np.random.normal(0, noise_amt * normalization, X_te_orig.shape).astype(np.double)
		X_te_1 = (X_te_orig + noise).astype(np.double)
		y_te_1 = y_te_orig.copy()
	elif shift == 'large_gn_shift_0.5':
		print('Large GN shift 0.5')
		normalization = 255.0
		X_te_1, _ = gaussian_noise_subset(X_te_orig, 100.0, normalization=normalization, delta_total=0.5)
		y_te_1 = y_te_orig.copy()
	elif shift == 'medium_gn_shift_0.5':
		print('Medium GN Shift 0.5')
		normalization = 255.0
		X_te_1, _ = gaussian_noise_subset(X_te_orig, 10.0, normalization=normalization, delta_total=0.5)
		y_te_1 = y_te_orig.copy()
	elif shift == 'small_gn_shift_0.5':
		print('Small GN Shift 0.5')
		normalization = 255.0
		X_te_1, _ = gaussian_noise_subset(X_te_orig, 1.0, normalization=normalization, delta_total=0.5)
		y_te_1 = y_te_orig.copy()
	elif shift == 'large_gn_shift_0.1':
		print('Large GN shift 0.1')
		normalization = 255.0
		X_te_1, _ = gaussian_noise_subset(X_te_orig, 100.0, normalization=normalization, delta_total=0.1)
		y_te_1 = y_te_orig.copy()
	elif shift == 'medium_gn_shift_0.1':
		print('Medium GN Shift 0.1')
		normalization = 255.0
		X_te_1, _ = gaussian_noise_subset(X_te_orig, 10.0, normalization=normalization, delta_total=0.1)
		y_te_1 = y_te_orig.copy()
	elif shift == 'small_gn_shift_0.1':
		print('Small GN Shift 0.1')
		normalization = 255.0
		X_te_1, _ = gaussian_noise_subset(X_te_orig, 1.0, normalization=normalization, delta_total=0.1)
		y_te_1 = y_te_orig.copy()
	elif shift == 'adversarial_shift_1.0':
		print('Large adversarial shift 1.0')
		# datset = datset[:5]
		adv_samples, true_labels = adversarial_samples(datset)
		X_te_1, y_te_1, _ = data_subset(X_te_orig, y_te_orig, adv_samples, true_labels, delta=1.0)
	elif shift == 'adversarial_shift_0.5':
		print('Medium adversarial Shift 0.5')
		adv_samples, true_labels = adversarial_samples(datset)
		X_te_1, y_te_1, _ = data_subset(X_te_orig, y_te_orig, adv_samples, true_labels, delta=0.5)
	elif shift == 'adversarial_shift_0.1':
		print('Small adversarial Shift 0.1')
		adv_samples, true_labels = adversarial_samples(datset)
		X_te_1, y_te_1, _ = data_subset(X_te_orig, y_te_orig, adv_samples, true_labels, delta=0.1)
	elif shift == 'ko_shift_0.1':
		print('Small knockout shift 0.1')
		X_te_1, y_te_1 = knockout_shift(X_te_orig, y_te_orig, 0, 0.1)
	elif shift == 'ko_shift_0.5':
		print('Medium knockout shift 0.5')
		X_te_1, y_te_1 = knockout_shift(X_te_orig, y_te_orig, 0, 0.5)
	elif shift == 'ko_shift_1.0':
		print('Large knockout shift 1.0')
		X_te_1, y_te_1 = knockout_shift(X_te_orig, y_te_orig, 0, 1.0)
	elif shift == 'small_img_shift_0.1':
		print('Small image shift 0.1')
		X_te_1, _ = image_generator(X_te_orig, orig_dims, 10, 0.05, 0.05, 0.1, 0.1, False, False, delta=0.1)
		y_te_1 = y_te_orig.copy()
	elif shift == 'small_img_shift_0.5':
		print('Small image shift 0.5')
		X_te_1, _ = image_generator(X_te_orig, orig_dims, 10, 0.05, 0.05, 0.1, 0.1, False, False, delta=0.5)
		y_te_1 = y_te_orig.copy()
	elif shift == 'small_img_shift_1.0':
		print('Small image shift 1.0')
		X_te_1, _ = image_generator(X_te_orig, orig_dims, 10, 0.05, 0.05, 0.1, 0.1, False, False, delta=1.0)
		y_te_1 = y_te_orig.copy()
	elif shift == 'medium_img_shift_0.1':
		print('Medium image shift 0.1')
		X_te_1, _ = image_generator(X_te_orig, orig_dims, 40, 0.2, 0.2, 0.2, 0.2, True, False, delta=0.1)
		y_te_1 = y_te_orig.copy()
	elif shift == 'medium_img_shift_0.5':
		print('Medium image shift 0.5')
		X_te_1, _ = image_generator(X_te_orig, orig_dims, 40, 0.2, 0.2, 0.2, 0.2, True, False, delta=0.5)
		y_te_1 = y_te_orig.copy()
	elif shift == 'medium_img_shift_1.0':
		print('Medium image shift 1.0')
		X_te_1, _ = image_generator(X_te_orig, orig_dims, 40, 0.2, 0.2, 0.2, 0.2, True, False, delta=1.0)
		y_te_1 = y_te_orig.copy()
	elif shift == 'large_img_shift_0.1':
		print('Large image shift 0.1')
		X_te_1, _ = image_generator(X_te_orig, orig_dims, 90, 0.4, 0.4, 0.3, 0.4, True, True, delta=0.1)
		y_te_1 = y_te_orig.copy()
	elif shift == 'large_img_shift_0.5':
		print('Large image shift 0.5')
		X_te_1, _ = image_generator(X_te_orig, orig_dims, 90, 0.4, 0.4, 0.3, 0.4, True, True, delta=0.5)
		y_te_1 = y_te_orig.copy()
	elif shift == 'large_img_shift_1.0':
		print('Large image shift 1.0')
		X_te_1, _ = image_generator(X_te_orig, orig_dims, 90, 0.4, 0.4, 0.3, 0.4, True, True, delta=1.0)
		y_te_1 = y_te_orig.copy()
	elif shift == 'medium_img_shift_0.5+ko_shift_0.1':
		print('Medium image shift 0.5 + knockout shift 0.1')
		X_te_1, _ = image_generator(X_te_orig, orig_dims, 40, 0.2, 0.2, 0.2, 0.2, True, False, delta=0.5)
		y_te_1 = y_te_orig.copy()
		X_te_1, y_te_1 = knockout_shift(X_te_1, y_te_1, 0, 0.1)
	elif shift == 'medium_img_shift_0.5+ko_shift_0.5':
		print('Medium image shift 0.5 + knockout shift 0.5')
		X_te_1, _ = image_generator(X_te_orig, orig_dims, 40, 0.2, 0.2, 0.2, 0.2, True, False, delta=0.5)
		y_te_1 = y_te_orig.copy()
		X_te_1, y_te_1 = knockout_shift(X_te_1, y_te_1, 0, 0.5)
	elif shift == 'medium_img_shift_0.5+ko_shift_1.0':
		print('Medium image shift 0.5 + knockout shift 1,0')
		X_te_1, _ = image_generator(X_te_orig, orig_dims, 40, 0.2, 0.2, 0.2, 0.2, True, False, delta=0.5)
		y_te_1 = y_te_orig.copy()
		X_te_1, y_te_1 = knockout_shift(X_te_1, y_te_1, 0, 1.0)
	elif shift == 'only_zero_shift+medium_img_shift_0.1':
		print('Only zero shift + Medium image shift 0.1')
		X_te_1, y_te_1 = only_one_shift(X_te_orig, y_te_orig, 0)
		X_te_1, _ = image_generator(X_te_1, orig_dims, 40, 0.2, 0.2, 0.2, 0.2, True, False, delta=0.1)
	elif shift == 'only_zero_shift+medium_img_shift_0.5':
		print('Only zero shift + Medium image shift 0.5')
		X_te_1, y_te_1 = only_one_shift(X_te_orig, y_te_orig, 0)
		X_te_1, _ = image_generator(X_te_1, orig_dims, 40, 0.2, 0.2, 0.2, 0.2, True, False, delta=0.5)
	elif shift == 'only_zero_shift+medium_img_shift_1.0':
		print('Only zero shift + Medium image shift 1.0')
		X_te_1, y_te_1 = only_one_shift(X_te_orig, y_te_orig, 0)
		X_te_1, _ = image_generator(X_te_1, orig_dims, 40, 0.2, 0.2, 0.2, 0.2, True, False, delta=1.0)
	elif shift == 'constant_shift':
		X_te_1, y_te_1 = (X_te_orig - 1.0, y_te_orig - 1.0)
	elif shift == 'bg_shift':
		X_te_1 = X_te_orig.copy()
		X_te_1[X_te_orig < 0.1] = 0.1  # np.random.sample(X_te_orig.shape)[X_te_orig < 0.1]
		y_te_1 = y_te_orig.copy()
	return (X_te_1, y_te_1)
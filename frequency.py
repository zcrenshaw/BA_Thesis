import h5py
import numpy as np

data_root = '/share/data/asl-data/fsvid/'
mat_paths = [ data_root + 'image128_color0_andy.mat'
                 , data_root + 'image128_color0_drucie.mat'
                 , data_root + 'image128_color0_rita.mat'
                 , data_root + 'image128_color0_robin.mat']

freqs = np.zeros(26)

for p in mat_paths:
	with h5py.File(p,'r') as f:
		for i in range(len(f['L'])):
			label = int(f['L'][i][0])
			if label >= 0 and label < 26:
				freqs[label] += 1
print(freqs)


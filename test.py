import master
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import sys
import numpy as np
import time

# path = '../testdcm/'
path = '../Dicom/'

Dicom = master.MRImage()
Dicom.loadDicom(path)

R1_img = Dicom.SI2R1()

C_a_time, C_a = np.loadtxt('../src/Data/Aorta.txt', skiprows=1, delimiter=';', unpack=True)

model = master.SingleSignalModelling()
model.set_time(Dicom.time)
model.C_a.insert_from_array(C_a, C_a_time)
model.C_a.downSample()


nrows = R1_img[0][0].shape[0]
ncols = R1_img[0][0].shape[1]
model_params_array = {'AIF': model.C_a(), 'params': [[[[] for _ in range(ncols)] for _ in range(nrows)] for _ in range(len(R1_img))]}


pbar = master.ProgressBar('Calculating (slice 1/{})'.format(len(R1_img)), len(R1_img)*nrows*ncols, show=True)
pbar_counter = 0

model.Model.make_lstsq_matrix_ETM()

for sl in range(len(R1_img)):
	for i in range(nrows):
		for j in range(ncols):
			signal = R1_img[sl][:,i,j]
			model.C_t.insert_from_array(signal)
			
			model.Model.update_lstsq_matrix_ETM_C_t()

			model_params = model.Model.fit_ETM()

			model_params_array['params'][sl][i][j] = {'model': 'ETM', 'params': model_params}
			pbar_counter = pbar.update(pbar_counter)

	pbar_counter = pbar.update(pbar_counter-1, 'Calculating (slice {}/{})'.format(sl+1,len(R1_img)))
pbar.finish()


shape = (len(model_params_array['params']), 1, len(model_params_array['params'][0][0]), len(model_params_array['params'][0][:]))
k_ep_img = np.zeros(shape)
K_trans_img = np.zeros(shape)

for k in range(k_ep_img.shape[0]):
	for i in range(k_ep_img.shape[2]):
		for j in range(k_ep_img.shape[3]):
			k_ep_img[k][0][i,j] = model_params_array['params'][k][i][j]['params']['k_ep']
			K_trans_img[k][0][i,j] = model_params_array['params'][k][i][j]['params']['K_trans']



fig = plt.figure()
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax4 = fig.add_subplot(2,2,4)


display = master.DisplayFunctions()
display.imshow(Dicom.slices, ax=ax1, time=Dicom.time)
display.imshow(R1_img, ax=ax2, time=Dicom.time, model_data_dict=model_params_array)
display.toolbuttons.activate_on(fig)

display.imshow(k_ep_img, ax=ax3, cmap='hot')
display.imshow(K_trans_img, ax=ax4, cmap='hot')

plt.show()

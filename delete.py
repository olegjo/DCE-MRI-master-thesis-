import master
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import sys
import numpy as np

path = '../Dicom/'


C_a_time, C_a = np.loadtxt('../src/Data/Aorta.txt', skiprows=1, delimiter=';', unpack=True)

model = master.MRImageAnalysis()
model.loadDicom(path)

model.set_AIF(C_a, C_a_time)

model.calculate('ETM', [0,1,2])

K_trans_img = model.getImage('K_trans')
k_ep_img    = model.getImage('k_ep')

fig = plt.figure()
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax4 = fig.add_subplot(2,2,4)


display = master.DisplayFunctions()
display.imshow(model.slices, ax=ax1, time=model.time)
display.imshow(model.R1_img, ax=ax2, time=model.time, model_data_dict=model.model_params_dict)
display.toolbuttons.activate_on(fig)

display.imshow(k_ep_img, ax=ax3, cmap='hot')
display.imshow(K_trans_img, ax=ax4, cmap='hot')

plt.show()

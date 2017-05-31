import dicom
import os
import sys
import numpy as np

# importing matplotlib-stuff
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import progressbar
import time


class SingleSignalModelling:
	class Concentration:
		def __init__(self, parent):
			self.parent = parent

		def __call__(self):
			return self.C

		def _insert(self, C, time):
			self.C_original = C.copy()
			self.C          = C
			self.time       = time

		def insert_from_array(self, C, time=None):
			if time is None:
				time = np.arange(len(C))
			self._insert(C, time)

		def addNoise(self, SNR):
			self._noise(SNR, 'C')

		def changeNoise(self, SNR):
			self._noise(SNR, 'C_original')

		def _noise(self, SNR, C_name):
			'''
				The noise is here modelled using a gaussian disatribution
				with standard deviation, sigma, given by 
							SNR = avg_signal/sigma
				and mean of zero.
			'''
			C          = getattr(self, C_name)
			avg_signal = np.mean(C)
			std        = abs(avg_signal/float(SNR))
			self.C     = C + np.random.normal(0, std, len(C))


		def downSample(self, C=None, old_time=None, new_time=None):
			'''
				Function to downsample one signal with timepoints
				old_time, into a new signal with time_points new_time
			'''
			if C is None:
				C = self.C
			if old_time is None:
				old_time = self.time
			if new_time is None:
				new_time = self.parent.time

			if new_time[-1] > old_time[-1] or new_time[0] < old_time[0]:
				print 'WARNING: new time must be a subset of old time.'
				return self.C

			new_C = np.zeros(len(new_time))
			for i in range(len(new_time)):
				time_diff = old_time - new_time[i]
				j = np.argmin(abs(time_diff))
				if time_diff[j] < 0:
					j1 = j-1
					j2 = j
				else:
					j1 = j
					j2 = j+1

				print j2, j1

				a = (C[j2] - C[j1])/(old_time[j2] - old_time[j1])
				b = -a*old_time[j1] + C[j1]

				new_C[i] = a*new_time[i] + b

			self.time = new_time
			self.C    = new_C
			return new_C


	class ArterialInputFunction(Concentration):
		Hct = 0.45
		def set_hematocrit(self, Hct):
			self.Hct = Hct

		def get_gamma_variate(self, t_0, A, alpha, beta):
			t = parent.time
			aif = A*(t - t_0)**alpha*np.exp(-(t - t_0)/beta)
			aif = np.nan_to_num(aif)
			return aif*(1 - self.Hct)

		def insert_gamma_variate(self, t_0, A, alpha, beta):
			self.C = self.get_gamma_variate(t_0, A, alpha, beta)

		def get_from_file(self, filename=None):
			if filename is None:
				filename = 'Data/Aorta.txt'
			t_C_a, C_a = np.loadtxt(filename, skiprows=1, delimiter=';', unpack=True)
			return t_C_a, C_a
			

		def insert_from_file(self, filename=None):
			t_C_a, C_a = self.get_from_file(filename)
			self._insert(C_a, t_C_a)


	class TissueConcentration(Concentration):
		def get_from_model(self, model, params):
			parent              = self.parent
			local_params        = params.copy()
			local_params['C_a'] = parent.C_a()
			local_params['t']   = parent.time

			attributes = {
				'tm'    : 'TM',
				'etm'   : 'ETM',
				'2cxm'  : 'twoCXM',
				'twocxm': 'twoCXM'
			}
			return getattr(parent.Models, attributes[model.lower()])(local_params), parent.time
			
		def insert_using_model(self, model, params):
			C, t = self.get_from_model(model, params)
			self._insert(C, t)

	class Models:
		def __init__(self, parent=None):
			self.integration_method = 'trapezoidal'
			self.parent             = parent
			self.A                  = None

		@staticmethod
		def TM(args):
			# extract all the parmeters stored in args.
			for var in args:
				globals()[var] = args[var]

			# compute the tissue concentration based on the Tofts model
			tofts = K_trans*np.exp(-t*k_ep)
			return NP.convolve(tofts, C_a, t)

		@staticmethod
		def ETM(args):
			for var in args:
				globals()[var] = args[var]


			tofts = K_trans*np.exp(-t*k_ep)
			return NP.convolve(tofts, C_a, t) + C_a*v_p
		
		@staticmethod
		def twoCXM(args):
			for var in args:
				globals()[var] = args[var]

			E  = PS/float(PS + F_p)
			e  = v_e/float(v_e + v_p)
			Ee = E*e
			
			tau_pluss = (E - Ee + e)/(2.*E)*(1 + np.sqrt(1 - 4*(Ee*(1-E)*(1-e))/(E - Ee + e)**2 ) )
			tau_minus = (E - Ee + e)/(2.*E)*(1 - np.sqrt(1 - 4*(Ee*(1-E)*(1-e))/(E - Ee + e)**2 ) )
			
			F_pluss = F_p*(tau_pluss - 1.)/(tau_pluss - tau_minus)
			F_minus = F_p*(tau_minus - 1.)/(tau_pluss - tau_minus)

			K_pluss = F_p/((v_p + v_e) * tau_minus)
			K_minus = F_p/((v_p + v_e) * tau_pluss)

			two_compartment_model = F_pluss*np.exp(-t*K_pluss) + F_minus*np.exp(-t*K_minus)
			return NP.convolve(two_compartment_model, C_a, t)
		
		def get_model_fit(self):
			params = self.model_params
			return getattr(self, self.model_used)(params)

		def initiate_lstsq_matrix_TM(self):
			self.A       = np.zeros((len(self.parent.C_a())-1, 2))
			self.A[:, 0] = self.cumulativeIntegral(self.parent.time, self.parent.C_a())

		def update_lstsq_matrix_TM(self):
			self.A[:, 1] = -self.cumulativeIntegral(self.parent.time, self.parent.C_t())

		def make_lstsq_matrix_TM(self):
			self.A       = np.zeros((len(self.parent.C_a())-1, 2))
			self.A[:, 0] = self.cumulativeIntegral(self.parent.time, self.parent.C_a())
			self.A[:, 1] = -self.cumulativeIntegral(self.parent.time, self.parent.C_t())


		def initiate_lstsq_matrix_ETM(self):
			self.A       = np.zeros((len(self.parent.C_a()) - 1, 3))
			self.A[:, 0] = self.cumulativeIntegral(self.parent.time, self.parent.C_a()) 
			self.A[:, 2] = self.parent.C_a()[1:]

		def update_lstsq_matrix_ETM(self):
			self.A[:, 1] = -self.cumulativeIntegral(self.parent.time, self.parent.C_t()) 

		def make_lstsq_matrix_ETM(self):
			self.A       = np.zeros((len(self.parent.C_a()) - 1, 3))
			self.A[:, 0] = self.cumulativeIntegral(self.parent.time, self.parent.C_a()) 
			self.A[:, 2] = self.parent.C_a()[1:]
			self.A[:, 1] = -self.cumulativeIntegral(self.parent.time, self.parent.C_t()) 

		
		def fit_TM(self, t=None, C_t=None, C_a=None):
			self.model_used = 'TM'
			if None in [t, C_t, C_a] or self.parent is not None:
				t   = self.parent.time
				C_t = self.parent.C_t()
				C_a = self.parent.C_a()
			if self.A is None:
				self.make_lstsq_matrix_TM()

			B       = np.linalg.lstsq(self.A, C_t[1:])
			k_ep    = B[0][1]
			K_trans = B[0][0]

			self.model_params = {'k_ep': k_ep, 'K_trans': K_trans}

			return self.model_params

		def fit_ETM(self):
			self.model_used = 'ETM'
			if self.A is None:
				self.make_lstsq_matrix_ETM()

			t   = self.parent.time
			C_t = self.parent.C_t()
			C_a = self.parent.C_a()

			B = np.linalg.lstsq(self.A, C_t[1:])

			k_ep    = B[0][1]
			v_p     = B[0][2]
			K_trans = B[0][0] - k_ep*v_p
			v_e     = K_trans/k_ep

			self.model_params = {'k_ep': k_ep, 'K_trans': K_trans, 'v_p': v_p, 'v_e': v_e}
			return self.model_params

		def fit_2CXM(self, t=None, C_t=None, C_a=None):
			#self.model_used = '2CXM'
			raise NotImplementedError, 'The two-compartment exchange model is not implemented yet.'

		def cumulativeIntegral(self, x, y):
			# getattr gets the method from this class. I.e if
			# self.integration_method = trapezoidal, the line below
			# will call the trapezoidal function.
			return getattr(self, self.integration_method)(x, y)

		@staticmethod
		def trapezoidal(x, y):
			return np.cumsum(0.5*(y[:-1] + y[1:])*(x[1:] - x[:-1]))


	def __init__(self, args=None):
		self.C_a = self.ArterialInputFunction(self)
		self.C_t = self.TissueConcentration(self)
		self.t   = None

		self.Model = self.Models(self)

		# this next bit is for an alternative call signature.
		if args is not None:
			fncs = {
				'time': 'self.set_time(args["time"])'
			}
			for key in ['C_a', 'C_t']:
				if key in args:
					fncs[key] = 'self.'+key+'.'+args[key]['method']+'(**'
					if 'args' in args[key]:
						fncs[key] += str(args[key]['args'])
					else:
						fncs[key] += str({})
					fncs[key] += ')'


			# need to be done in a specific order
			fncs_keys = ['C_a', 'time', 'C_t']
			for key in fncs_keys:
				if key in args:
					eval(fncs[key])



	def set_time(self, t):
		if str(t) == 'C_a':
			self.t = self.C_a.time
			return
		self.t = t

	@property
	def dt(self):
		return self.t[1] - self.t[0]

	@property
	def time(self):
		if self.t is None:
			raise AttributeError('Time not set.')
		return self.t

	def fit_to_model(self, model):
		return getattr(self.Model, 'fit_'+model)()

class MRImage:
	def __init__(self):
		self.slices         = None
		self.slices_dicom   = None
		self.time           = None
		self.sliceLocations = None
		self.R1_img         = None
		
		self.isDicom        = False

	@property
	def nSlices(self):
		return len(self.slices)

	@property
	def nTimePoints(self):
		return len(self.slices[0])



	def loadDicom(self, path, load=None):
		'''
			Loads dicom files and stores their pixel arrays,
			ordered in slice location and time point.
			
			args
				path: path to directory containing dicom files
				load: list of indeces of files to load

			returns
				4D array of the pixel arrays:
				ARRAY[sliceIdx][timeIdx,ypos,xpos]
		'''
		self.isDicom = True

		# find available dicom files
		lstFilesDCM = []
		for dirName, subdirList, fileList in os.walk(path):
			for filename in fileList:
				if '.dcm' in filename.lower():
					lstFilesDCM.append(os.path.join(dirName, filename))

		if load is None:
			lstFilesDCM = np.array(lstFilesDCM)
		elif type(load) is list:
			lstFilesDCM = np.array(lstFilesDCM)[load]
		else:
			raise RuntimeWarning('Load must be of type list. Used all dicom files.')

		# The dicom objects are not necessarily ordered correcty
		# Place all dcm objects in a list for further processing
		DicomList = []
		pbar = ProgressBar('Loading dicom...', len(lstFilesDCM))
		for filename in lstFilesDCM:
			DicomList.append(dicom.read_file(filename))
			pbar.update()
		pbar.finish()
		# the next bit is to extract all the dicom files.
		# to be sure, I also sort the arrays in order of 
		# slice location, and aquisition time.
		sliceDict = {
			'slices'        : [],
			'slices_dicom'  : [],
			'time'          : [],
			'sliceLocations': []
		}

		curr_time = 0 # for use if the dicom objects do not contain time information
		dt        = 3400
		if not hasattr(DicomList[0], 'TriggerTime'):
			print ('WARNING: No time info found in Dicom. Using 3.4s time resolution')
		
		for dcm in DicomList:
			try:
				sliceLoc = float(dcm.SliceLocation)
			except AttributeError:
				sliceLoc = 0

			if not sliceLoc in sliceDict['sliceLocations']:
				sliceDict['sliceLocations'].append(sliceLoc)
				for key in ['slices', 'slices_dicom', 'time']:
					sliceDict[key].append([])

			# find the index of the current slice location
			idx = np.where(np.array(sliceDict['sliceLocations']) == sliceLoc)[0][0]
			# append the slice data to the respective places:
			
			sliceDict['slices_dicom'][idx].append(dcm)
			sliceDict['slices'][idx].append(dcm.pixel_array)

			try:
				sliceDict['time'][idx].append(float(dcm.TriggerTime))
			except AttributeError:
				curr_time += dt
				sliceDict['time'][idx].append(curr_time)

		# make the lists numpy arrays
		for key in sliceDict:
			sliceDict[key] = np.array(sliceDict[key])

		# sort the slices in order of slicelocation
		argsort = np.argsort(sliceDict['sliceLocations'])
		for key in sliceDict:
			sliceDict[key] = sliceDict[key][argsort]

		# check is all time entries are equal
		equal_time = 1
		for i in range(len(sliceDict['time'])):
			equal_time *= np.array_equal(sliceDict['time'][0], sliceDict['time'][i])
		if not equal_time:
			raise RuntimeWarning('There seems to be a problem with dicom time. Using time of first slice.')


		# Order the time points of each slice in ascending order
		sliceDict['time'] = sliceDict['time'][0] - sliceDict['time'][0][0]
		argsort           = np.argsort(sliceDict['time'])
		sliceDict['time'] = sliceDict['time'][argsort]
		for key in ['slices', 'slices_dicom']:
			for i in range(len(sliceDict[key])):
				sliceDict[key][i] = sliceDict[key][i][argsort]

		self.slicelocations = sliceDict['sliceLocations']
		self.time           = sliceDict['time']/1000. # time in seconds
		self.slices         = sliceDict['slices']
		self.slices_dicom   = sliceDict['slices_dicom']

		return sliceDict['slices']

	def SI2R1(self, r_1=4.5, T_10=1.4, FA=None, TR=None, force_start=False):
		'''
			Converts an image, or a set of images, from signal
			intensity to relaxation rate.
			The values
				r1   = 4.5 s^-1 mM^-1
				T_10 = 1.4 s
			are commonly used values [1].

			We have
				R_1 = R_10 + r_1C
			and
				S = S_0 (1 - E_1)sin(FA) / (1 - E_1 sin(FA))
				E_1 = exp(-TR/T1)
				FA = flip angle
				TR = repetition time

			[1] Tofts, Paul S. "T1-weighted DCE imaging concepts: modelling, acquisition and analysis." signal 500.450 (2010): 400.
		'''
		if self.R1_img is not None and not force_start:
			print 'R1 image already obtained. To force start the process, please pass the argument force_start=True.'
			return self.R1_img

		if not self.isDicom and None in [FA, TR]:
			print 'Flip angle (FA) and/or repetition time (TR) not supplied. Please enter the values.'
			FA = float(raw_input('FA = '))
			TR = float(raw_input('TR = '))
		elif self.isDicom:
			FA = self.slices_dicom[0][0].FlipAngle*np.pi/180. # in radians
			TR = self.slices_dicom[0][0].RepetitionTime/1000. # in seconds

		self.R1_img     = np.zeros(self.slices.shape)
		E_10            = np.exp(-TR/T_10)
		B               = (1. - E_10)/(1. - E_10*np.cos(FA))
		cosFA           = np.cos(FA)
		minus_1_over_TR = -1./TR
		
		pbar         = ProgressBar('SI to R1 (slice 1/{})'.format(self.nSlices), self.nSlices)
	
		for sl in range(self.nSlices):
			S = self.slices[sl]
			S_0 = np.mean(S[:5], axis=0)
			A = np.nan_to_num(B*S/S_0)
			self.R1_img[sl] = np.nan_to_num(minus_1_over_TR*np.log((1. - A)/(1. - A*cosFA)))
			
			pbar.update(0, 'SI to R1 (slice {}/{})'.format(sl+1, self.nSlices))
		pbar.finish()
		self.R1_img -= np.average(self.R1_img[:5])

		return self.R1_img

class DisplayFunctions:
	def __init__(self):
		self.toolbuttons = self.ToolButtons()

		self.timePlotParams = {
			'xlabel': 'time (s)',
			'ylabel': 'Signal Intensity'
		}

	def imshow(self, data_array, showfig=False, ax=None, time=None, model_data_dict=None, cmap='gray'):
		'''
			plots and shows data in a 5D array, data_array.
			Should contain images arranged by slices, then time points

			args:
				data_array[slice][time_point][ypos, xpos] (or data_array[slice, time_point, ypos, xpos])
				show (bool): should the figure be shown immidiately?
				ax (matplotlib.pyplot.Axes): axes object to draw the figure on
				time (array like): list of time points
				model_data_dict (dict): dictionary of model attributes.
					model_data_dict[slice][ypos][xpos]['AIF']: arterial input function
					model_data_dict[slice][ypos][xpos]['params']['model']: model used
					model_data_dict[slice][ypos][xpos]['params']['params']: model params
		'''
		if ax is None:
			# initiate figure
			fig = plt.figure()
			ax  = fig.add_subplot(111)

		self.mainfigure = ax.figure
		self.mainax = ax
		self.time = time
		self.model_data_dict = model_data_dict

		ax.set_xticks([])
		ax.set_yticks([])
		plt.subplots_adjust(left=0.25, bottom=0.25)
		ax.time = time
		ax.data_array = data_array
		ax.currentSliceIdx = 0


		# initiate the image
		image = ax.imshow(data_array[0,0], cmap=plt.get_cmap(cmap), interpolation='nearest')#, vmin=np.min(data_array), vmax=np.max(data_array))

		# add sliders to move through slices and timepoints
		sliceAx = sliceAx = inset_axes(ax,
					width="100%",  # width = 10% of parent_bbox width
					height="5%",  # height : 50%
					loc=3,
					bbox_to_anchor=(0., -.1, 1, 1),
					bbox_transform=ax.transAxes,
					borderpad=0
					)
		sliceSlider = Slider(sliceAx, 'Slice', 0, len(data_array)-1, valinit=0, valfmt='%i')

		timeAx = inset_axes(ax,
					width="100%",  # width = 10% of parent_bbox width
					height="5%",  # height : 50%
					loc=3,
					bbox_to_anchor=(0., -.17, 1, 1),
					bbox_transform=ax.transAxes,
					borderpad=0
					)
		timeSlider = Slider(timeAx, 'Time', 0, len(data_array[0])-1, valinit=0, valfmt='%i')
		if len(data_array)-1 == 0:
			sliceAx.set_visible(False)
		if len(data_array[0]) == 0:
			timeAx.set_visible(False)

		# global variables to store the current slice and time index
		currentSliceIdx = 0
		currentTimeidx  = 0

		def update(val):
			global currentTimeidx; global currentSliceIdx
			valTime  = int(timeSlider.val)
			valSlice = int(sliceSlider.val)

			if valTime >= len(data_array[0]):
				currentTimeidx = len(data_array[0]-1)
			else:
				currentTimeidx = valTime

			if valSlice >= len(data_array):
				currentSliceIdx = len(data_array)-1
			else:
				currentSliceIdx = valSlice

			ax.currentSliceIdx = currentSliceIdx

			image.set_data(data_array[currentSliceIdx, currentTimeidx])
			ax.draw_artist(image)

		def plotTimeCurve(event):
			fig = plt.figure()
			ax  = fig.add_subplot(111)
			eventax = event.inaxes
			xpos, ypos = int(event.xdata), int(event.ydata)
			ydata = eventax.data_array[eventax.currentSliceIdx][:,ypos,xpos]
			xdata = eventax.time

			ax.set_xlabel(self.timePlotParams['xlabel'])
			ax.set_ylabel(self.timePlotParams['ylabel'])

			if xdata is None:
				xdata = np.arange(len(ydata))
				ax.set_xlabel('Time info not found')
			ax.plot(xdata, ydata, label='Signal')
			return fig, ax

		def plotModel(event, model_data_dict, ax=None):
			if ax is None:
				fig = plt.figure()
				ax  = fig.add_subplot(111)

			
			xpos, ypos   = int(event.xdata), int(event.ydata)
			if event.inaxes.currentSliceIdx < len(model_data_dict['params']):
				model        = model_data_dict['params'][event.inaxes.currentSliceIdx][ypos][xpos]['model'].lower()
				model_params = model_data_dict['params'][event.inaxes.currentSliceIdx][ypos][xpos]['params']

				if model == 'tm':
					model = 'TM'
				elif model == 'etm':
					model = 'ETM'
				elif model in ['2cxm', 'twocxm']:
					model = 'twoCXM'

				model_params['C_a'] = model_data_dict['AIF']
				x = event.inaxes.time
				model_params['t'] = x
				y = getattr(SingleSignalModelling.Models, model)(model_params)

				ax.plot(x, y, label='Model fit')
				ax.plot(x, model_data_dict['AIF'], label='AIF')
				ax.legend()
				#ax.text(1, 1, 'data attributes')
				for key in model_params:
					print key, model_params[key]


		def clickFuncs(ax):
			def on_press(event):
				if event.inaxes in [ax]:
					if self.toolbuttons.activeButton is not None:
						if self.toolbuttons.activeButton.lower() == 'plot':
							if None not in [event.xdata, event.ydata]:
								fig1, ax1 = plotTimeCurve(event)
								if model_data_dict is not None:
									plotModel(event, model_data_dict, ax=ax1)
								plt.show()
						if self.toolbuttons.activeButton.lower() == 'refresh':
							sliceSlider.on_changed(update)
							timeSlider.on_changed(update)
			ax.figure.canvas.mpl_connect('button_press_event', on_press)

		
		sliceSlider.on_changed(update)
		timeSlider.on_changed(update)
		clickFuncs(ax)
		
		if showfig:
			plt.show()

		return ax



	class ToolButtons:
		def __init__(self):
			self.activeButton = None

		def activate_on(self, fig, labels=None, callbacks=None):
			self.fig          = fig

			self.labels = ['None', 'Plot']
			if labels is not None:
				self.labels = labels

			self.buttons = []
			self.btnaxes = []
			for i in range(len(self.labels)):
				btnax = fig.add_axes([0.05, 0.85-0.11*i, 0.1, 0.1], aspect='equal')
				btn   = Button(btnax, self.labels[i])
				self.buttons.append(btn)
				self.btnaxes.append(btnax)
				
				btn.on_clicked(self.toggle)
				if callbacks is not None:
					btn.on_clicked(callbacks[i])
			if 'None' in self.labels:
				self.buttons[np.where(np.array(self.labels) == 'None')[0][0]].color='0.95'

			# for some reason the buttons only get clickable when i have this line...
			some_button_that_i_need_for_some_reason = self.fig.add_axes([0.81, 0.05, 0.1, 0.075]).set_visible(False)

		
		def toggle(self, event):
			# first make all buttons normal
			for btn in self.buttons:
				btn.color = '0.85'
				btn.hovercolor = '0.95'

			# find which button got clicked
			ax = event.inaxes
			btnIdx = None
			for i in range(len(self.btnaxes)):
				if ax == self.btnaxes[i]:
					btnIdx = i
					break

			# change background color on clicked button
			if btnIdx is not None:
				self.activeButton = self.labels[btnIdx]
				self.buttons[btnIdx].color = '0.95'

class MRImageAnalysis(MRImage):
	def __init__(self):
		MRImage.__init__(self)

	def set_time(self, time):
		self.time = time

	def set_AIF(self, C_a, C_a_time):
		self.C_a = C_a
		self.C_a_time = C_a_time

	def calculate(self, modelName, slices=None):
		if modelName.lower() == 'tm':
			modelName = 'TM'
		if modelName.lower() == 'etm':
			modelName = 'ETM'
		if modelName.lower() in ['2cxm', 'twocxm']:
			modelName = 'twoCXM'

		R1_img = self.SI2R1()

		model = SingleSignalModelling()
		model.set_time(self.time)
		model.C_a.insert_from_array(self.C_a, self.C_a_time)
		model.C_a.downSample()

		nrows = R1_img[0][0].shape[0]
		ncols = R1_img[0][0].shape[1]

		if slices is None or type(slices) != type([]):
			slices = range(len(R1_img))

		model_params_dict = {'AIF': model.C_a(), 'params': [[[[] for _ in range(ncols)] for _ in range(nrows)] for _ in range(len(slices))]}

		# initiate the matrix for least squares
		getattr(model.Model, 'initiate_lstsq_matrix_'+modelName)()


		# make a progressbar
		pbar = ProgressBar('Calculating slice 1/{}'.format(len(slices)), len(slices)*nrows*ncols)
		for sl in slices:
			for i in range(nrows):
				for j in range(ncols):
					# extract the time signal for the current pixel
					# and insert it into the model
					signal = R1_img[sl,:,i,j]

					model.C_t.insert_from_array(signal)

					# update the matrix for least squares using the new signal
					getattr(model.Model, 'update_lstsq_matrix_'+modelName)()

					# calculate the model fit
					model_params = getattr(model.Model, 'fit_'+modelName)()
					
					# store the fitting paramaters
					model_params_dict['params'][sl][i][j] = {'model': modelName, 'params': model_params}

					# update the progressbar
					pbar.update()
			pbar.update(-1, 'Calculating (slice {}/{})'.format(sl+1,len(slices)))
		pbar.finish()

		self.model_params_dict = model_params_dict

	def getImage(self, paramater_name):
		shape = (len(self.model_params_dict['params']), 1, len(self.model_params_dict['params'][0][0]), len(self.model_params_dict['params'][0][:]))
		image = np.zeros(shape)
		for k in range(image.shape[0]):
			for i in range(image.shape[2]):
				for j in range(image.shape[3]):
					image[k,0,i,j] = self.model_params_dict['params'][k][i][j]['params'][paramater_name]
		return image


class ProgressBar:
	def __init__(self, descriptor, maxval, widgets=None, show=True):
		self.maxval  = maxval
		self.widgets = widgets
		self.pbar_counter = 0
		if widgets is None and show:
			self.widgets = [' {:25.25}'.format(descriptor), ' ', progressbar.Bar(), ' ', progressbar.Percentage(), ' ', progressbar.ETA()]
		if not show:
			self.widgets = ['']
		self.pbar = progressbar.ProgressBar(maxval=self.maxval, widgets=self.widgets).start()

	def update(self, i=0, descriptor=None):
		self.pbar.update(self.pbar_counter+i)
		if descriptor is not None:
			self.widgets[0] = ' {:25.25}'.format(descriptor)
		self.pbar_counter += 1+i

	def finish(self):
		for i in range(len(self.widgets)):
			if type(self.widgets[i]) == type(progressbar.Percentage()):
				self.widgets.pop(i)
				self.widgets.pop(i)
				break
		self.pbar.finish()


class NP:
	'''
		Class with modified numpy functions to make code more dense
	'''
	@staticmethod
	def convolve(x, y, t):
		return (np.convolve(x, y)*(t[1]-t[0]))[:len(t)]











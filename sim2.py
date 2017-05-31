import master
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import numpy as np


###############################################
#####               SETTINGS              #####
###############################################

start_model = 'ETM'
end_model   = 'ETM'

show_animation = True

###############################################
###############################################




params = {
	'2CXM': {
		'PS' : 0.12/60,
		'F_p': 1.2/60,
		'v_p': 0.1,
		'v_e': 0.2
	},
	'ETM': {
		'k_ep'   : 0.0700,
		'K_trans': 0.0277,
		'v_p'    : 0.076197,
		'v_e'    : 0.396279
	},
	'ETM-1': {
		'v_p'    : 0.0285583247215,
		'k_ep'   : 0.00976500445893,
		'K_trans': 0.00400894139204,
		'v_e'    : 0.410541685762
	},
	'TM': {
		'k_ep'   : 0.0700,
		'K_trans': 0.0277
	}
}

def make_plots(start_model, end_model, params):
	model = master.SingleSignalModelling()
	model.C_a.insert_from_file()
	model.set_time(model.C_a.time)

	SNR = np.linspace(1, 40, 100)
	values = []
	err_created = False

	model.C_t.insert_using_model(start_model, params)
	for i in range(len(SNR)):
		model.C_t.changeNoise(SNR[i])
		getattr(model.Model, 'make_lstsq_matrix_'+end_model)()
		model_params = getattr(model.Model, 'fit_'+end_model)()
		vals = {
			'C_t'      : model.C_t(),
			'C_a'      : model.C_a(),
			'model_fit': model.Model.get_model_fit(),
			't'        : model.time,
			'SNR'      : SNR[i]
		}
		if not err_created:
			errs = np.zeros((len(model_params),len(SNR)))
			err_names = model_params.keys()
			err_created = True
		err = calculateError(params, model_params)
		for j in range(len(err_names)):
			if err_names[j] in params:
				errs[j,i] = err[err_names[j]]

		values.append(vals)

	return values, errs, err_names, SNR

def calculateError(true, computed):
	errors = {}
	for param_name in true:
		if param_name in computed:
			errors[param_name] = abs(computed[param_name] - true[param_name])/true[param_name]*100
	return errors


A, errs, err_names, SNR = make_plots(start_model, end_model, params[start_model])


fig = plt.figure()
plots = []
for i in range(len(err_names)):
	ax = fig.add_subplot(len(err_names),1,i+1)
	ax.plot(SNR, errs[i], label=err_names[i])
	ax.legend(loc='best')
	ax.set_xlabel('SNR')
	ax.set_ylabel('Error (%)')
	plots.append(ax)
plots[0].set_title('${} \\rightarrow {}$'.format(start_model, end_model))


if show_animation:
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	ax.set_xlabel('Time / s')
	ax.set_ylabel('Tissue concentration')

	ln1, = ax.plot([], [], 'b-', label='{}, ideal'.format(start_model))
	ln2, = ax.plot([], [], 'r-', label='{}, ideal'.format(end_model))
	ax.legend()

	pause = False
	def onClick(event):
		global pause
		pause ^= True

	iterator = 0
	def animate(j):
		global iterator, pause
		i = iterator
		if not pause:
			iterator += 1
			if iterator >= len(A):
				print 'Click anywhere to restart the animation'
				iterator = 0
				pause = True
			j = (-i+1)

			t = A[i]['t']
			y1 = A[i]['C_t']
			y2 = A[i]['model_fit']

			ln1.set_data(t, y1)
			ln2.set_data(t, y2)

			xmin = min(t)
			xmax = max(t)
			ymin = min([min(y1), min(y2)])
			ymax = max([max(y1), max(y2)])

			ax.set_xlim(xmin, xmax)
			ax.set_ylim(ymin, ymax)

			global title
			ax.set_title('SNR = {}'.format(A[i]['SNR']))

			return ln1, ln2

	print 'Click anywhere to pause the animation'
	fig.canvas.mpl_connect('button_press_event', onClick)

	ani = animation.FuncAnimation(fig, animate, interval=100, frames=range(len(A)), repeat=True)

plt.show()









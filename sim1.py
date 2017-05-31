import master
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation



###############################################
#####               SETTINGS              #####
###############################################

start_model = 'TM'
end_model   = 'ETM'

show_animation = True

###############################################
###############################################


def makePlots(start_model, end_model, params, C_a, C_a_time):
	dts = np.linspace(0.1, 10, 100)
	model = master.SingleSignalModelling()

	err_created = False
	values = []
	for i in range(len(dts)):
		dt = dts[i]
		t = C_a_time[0::int(dt*10)]

		model.set_time(t)
		model.C_a.insert_from_array(C_a[0::int(dt*10)], t)
		model.C_t.insert_using_model(start_model, params)
		getattr(model.Model, 'make_lstsq_matrix_'+end_model)()
		
		model_params = getattr(model.Model, 'fit_'+end_model)()
		vals = {
			'C_t'      : model.C_t(),
			'C_a'      : model.C_a(),
			'model_fit': model.Model.get_model_fit(),
			't'        : t
		}
		if not err_created:
			errs        = np.zeros((len(model_params), len(dts)))
			err_names   = model_params.keys()
			err_created = True

		err = calculateError(params, model_params)
		for j in range(len(err_names)):
			if err_names[j] in params:
				errs[j,i] = err[err_names[j]]
		values.append(vals)

	return values, errs, err_names, dts


def calculateError(true, computed):
	errors = {}
	for param_name in true:
		if param_name in computed:
			errors[param_name] = abs(computed[param_name] - true[param_name])/true[param_name]*100
	return errors



t_C_a, C_a = np.loadtxt('../src/Data/Aorta.txt', skiprows=1, delimiter=';', unpack=True)

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
	'TM': {
		'k_ep'   : 0.0700,
		'K_trans': 0.0277
	},
	'ETM-1': {
		'v_p'    : 0.0285583247215,
		'k_ep'   : 0.00976500445893,
		'K_trans': 0.00400894139204,
		'v_e'    : 0.410541685762
	}
}

A, errs, err_names, dts = makePlots(start_model, end_model, params[start_model], C_a, t_C_a)

fig = plt.figure()
plots = []
for i in range(len(err_names)):
	ax = fig.add_subplot(len(err_names),1,i+1)
	ax.plot(dts, errs[i], label=err_names[i])
	ax.legend(loc='best')
	ax.set_xlabel('$\Delta t$')
	ax.set_ylabel('Error (%)')
	plots.append(ax)
plots[0].set_title('${} \\rightarrow {}$'.format(start_model, end_model))


def animate(i):
	global xmin1
	global xmax1
	global ymin1
	global ymax1
	j = (-i+1)
	if i == 0:
		ln2.set_data(A[i]['t'], A[i]['C_t'])
		xmin1 = min(A[i]['t'])
		xmax1 = max(A[i]['t'])
		ymin1 = min(A[i]['C_t'])
		ymax1 = max(A[i]['C_t'])

	t = A[j]['t']
	y1 = A[j]['C_t']
	ln1.set_data(t, y1)

	global xmin
	global xmax
	global ymin
	global ymax
	# print 'x: ({}, {}), y: ({}, {})'.format(xmin, xmax, ymin, ymax)

	xmin = min(t)
	xmax = max(t)
	ymin = min([ymin, min(y1), ymin1])
	ymax = max([ymax, max(y1), ymax1])

	ax.set_xlim(xmin, xmax)
	ax.set_ylim(ymin, ymax)

	ax.set_title('dt = {}'.format(t[1]-t[0]))

	return ln1


if show_animation:
	xmin = 10
	xmax = 0
	ymin = 10
	ymax = 0

	fig  = plt.figure()
	ax   = fig.add_subplot(111)
	ax.set_xlabel('Time / s')
	ax.set_ylabel('Signal')

	ln1, = ax.plot([], [], 'r-')
	ln2, = ax.plot([], [], 'b-')
	ax.legend()

	ani = animation.FuncAnimation(fig, animate, interval=100, frames=range(len(A)), repeat=False)

plt.show()

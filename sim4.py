from master import *

###############################################
#####               SETTINGS              #####
###############################################

start_model = '2CXM'
end_model   = 'ETM'

SNR = 6


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


N = 200 # number of simulations for each dt
dts = np.linspace(0.1, 20, 100)
measurements = {
	'k_ep': np.zeros((len(dts), 2)),
	'K_trans': np.zeros((len(dts), 2))
}

model = SingleSignalModelling()

t_, C_a_ = model.C_a.get_from_file()


for i in range(len(dts)):
	dt     = dts[i]
	t      = t_[0::int(dt*10)]
	C_a    = C_a_[0::int(dt*10)]
	dts[i] = t[1]-t[0]

	model.C_a.insert_from_array(C_a, t)
	model.set_time(t)
	model.C_t.insert_using_model(start_model, params[start_model])

	k_ep    = np.zeros(N)
	K_trans = np.zeros(N)

	for j in range(N):
		model.C_t.changeNoise(SNR)
		model.Model.make_lstsq_matrix_ETM()
		model_params = model.fit_to_model(end_model)

		k_ep[j]      = model_params['k_ep']
		K_trans[j]   = model_params['K_trans']

	measurements['k_ep'][i]    = [np.mean(k_ep), np.std(k_ep)]
	measurements['K_trans'][i] = [np.mean(K_trans), np.std(K_trans)]


fig = plt.figure()
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)

ax1.errorbar(dts, measurements['k_ep'][:,0], yerr=measurements['k_ep'][:,1], label='Simulation')
ax1.set_ylabel('$k_\mathrm{ep}$')
plt.setp(ax1.get_xticklabels(), visible=False)

ax2.errorbar(dts, measurements['K_trans'][:,0], yerr=measurements['K_trans'][:,1], label='Simulation')
ax2.set_ylabel('$K^\mathrm{trans}$')
ax2.set_xlabel('$\Delta t$')

if start_model == 'ETM':
	ax1.plot(dts, np.zeros(len(dts))+params['ETM']['k_ep'], label='True')
	ax2.plot(dts, np.zeros(len(dts))+params['ETM']['K_trans'], label='True')

ax1.legend(loc='best')
ax2.legend(loc='best')

plt.show()












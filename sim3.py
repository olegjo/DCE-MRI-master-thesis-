from master import *
'''
	This program creates an ideal tissue concentration curve based
	on the Extended Tofts Model (ETM), adds noise to it, and creates a
	model fit based on it. The noise level is changed based on the SNR
	and for each SNR, a number of simultations are done. The average
	and std of the measured parameters are then recorded and plotted
	versus the SNR value.
'''


###############################################
#####               SETTINGS              #####
###############################################

start_model = '2CXM'
end_model   = 'ETM'

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


N = 100 # number of simulations

SNRs = np.linspace(1, 20, 40)

measurements = {
	'k_ep'   : np.zeros((len(SNRs), 2)),
	'K_trans': np.zeros((len(SNRs), 2))
}

model = SingleSignalModelling({
		'time': 'C_a',
		'C_a' : {
			'method': 'insert_from_file'
		},
		'C_t' : {
			'method': 'insert_using_model',
			'args': {
				'model' : start_model,
				'params': params[start_model]
			}
		}
})

for i in range(len(SNRs)):
	k_ep = np.zeros(N)
	K_trans = np.zeros(N)
	for j in range(N):
		model.C_t.changeNoise(SNRs[i])
		model_params = getattr(model.Model, 'fit_'+end_model)()
		k_ep[j] = model_params['k_ep']
		K_trans[j] = model_params['K_trans']
	measurements['k_ep'][i] = [np.mean(k_ep), np.std(k_ep)]
	measurements['K_trans'][i] = [np.mean(K_trans), np.std(K_trans)]
		

fig = plt.figure()
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)

ax1.errorbar(SNRs, measurements['k_ep'][:,0], yerr=measurements['k_ep'][:,1], label='Simumation')
ax1.set_ylabel('$K_\mathrm{ep}$')
plt.setp(ax1.get_xticklabels(), visible=False)

ax2.errorbar(SNRs, measurements['K_trans'][:,0], yerr=measurements['K_trans'][:,1], label='Simumation')
ax2.set_ylabel('$K^\mathrm{trans}$')
ax2.set_xlabel('SNR')

plt.show()























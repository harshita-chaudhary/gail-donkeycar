import glob
import numpy as np
import json

data_path = "/home/phoenix/mysim/data/"
save_path = 'expert_donkeycar_expert1'
actions = []
obs = []
catalog_files = glob.glob(data_path + '*.catalog')

for catalog_filename in catalog_files:
	f = open(catalog_filename, 'r')
	lines = f.readlines()
	for line in lines:
	   data = json.loads(line)
	   actions.append([data['user/angle'], data['user/throttle']])
	   obs.append(data_path + "images/"+ data['cam/image_array'])	   
	f.close()

   
actions = np.array(actions)
obs = np.array(obs)
    
numpy_dict = {
        'actions': actions,
        'obs': obs
}

for key, val in numpy_dict.items():
        print(key, val.shape)
        

if save_path is not None:
        np.savez(save_path, **numpy_dict)

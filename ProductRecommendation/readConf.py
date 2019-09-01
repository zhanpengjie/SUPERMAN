import json

# read confs from recommendation_conf.json and return it
def readConf(conf_path): 

	with open(conf_path,'r') as load_f:

		conf = json.load(load_f)

     		return conf
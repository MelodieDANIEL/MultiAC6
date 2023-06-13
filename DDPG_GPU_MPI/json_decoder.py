import json
import os

class JsonDecoder():
	
	def __init__(self, config_file_name=None):
		self.config_file = config_file_name
		
		self.config_dir_name = os.path.dirname(self.config_file) + '/'

		if not os.path.isfile(self.config_file):
			raise RuntimeError("=> Config file JSON to load does not exit : " + self.config_file)
			return 

		self.load_config()


	def load_config(self):
		with open(self.config_file, mode='r') as f:
			self.config_data = json.load(f)

	def print_config(self):
		print(self.config_data)
		
	"""	
	def has_key(self, key_name=None):
		return key_name in self.config_data
	"""

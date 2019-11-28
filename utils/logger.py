from utils.watch_time import time_str

class Logger:
	def __init__(self , inner_logger , fil_path = None):
		self.inner_logger = inner_logger
		if fil_path:
			self.inner_logger.add_file(path = fil_path)

	def nolog(self , cont = ""):
		pass

	def log_print(self , cont = ""):
		self.inner_logger.info(cont)

	def log_print_w_time(self , cont = ""):
		self.log_print(str(cont) + " | " + time_str())

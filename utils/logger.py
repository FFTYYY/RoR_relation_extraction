from utils.watch_time import time_str

class Logger:
	def __init__(self , fil_path = None):
		self.log_fil = open(fil_path , "w" , encoding = "utf-8")

	def nolog(self , cont = ""):
		print (cont)

	def log_print(self , cont = ""):
		self.log_fil.write(cont + "\n")
		self.log_fil.flush()
		print (cont)

	def log_print_w_time(self , cont = ""):
		self.log_print(str(cont) + " | " + time_str())

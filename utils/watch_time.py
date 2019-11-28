import time

starttime = time.time()

def now_time():
	return time.time() - starttime

def time_str():
	return "now time: %ds" % (now_time())

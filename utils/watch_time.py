import time
import fitlog

#fitlog.commit(__file__)

starttime = time.time()

def now_time():
	return time.time() - starttime

def time_str():
	return "now time: %ds" % (now_time())

__all__ = [now_time , time_str]
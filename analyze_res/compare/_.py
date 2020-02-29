import json
from pprint import pprint

with open("ace2005-wl.json" , "r") as f:
	a = json.load(f)

pprint (a[0])
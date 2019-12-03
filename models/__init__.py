from .naive_bert import Model as naive_bert
from .graph_trans import Model as graph_trans


models = {
	"naive_bert" : naive_bert , 
	"graph_trans" : graph_trans , 
}
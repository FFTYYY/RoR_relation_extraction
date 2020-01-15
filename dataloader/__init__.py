from .dataloader_semeval_2018_task7 import read_data as read_data_semeval_2018_task7
from .dataloader_ace05 import read_data as read_data_ace05


def get_dataloader(dataset_type):

	if dataset_type == "semeval_2018_task7":
		read_method = read_data_semeval_2018_task7
	if dataset_type == "ace_2005":
		read_method = read_data_ace05
	return read_method
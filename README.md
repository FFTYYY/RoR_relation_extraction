## How to Run

### Prepare data and environment
- ACE2005 dataset: You can download from its [LDC website](https://catalog.ldc.upenn.edu/LDC2006T06).

- SemEval2010 Task 7.2 dataset: You can download the publicly available data by the following command:
```
bash prep_data_and_env.sh
```
This command will also prepare the python environment for you. It will install all the packages in [requirements.txt](requirements.txt).

### Training

1. ACE2005:
`python main.py --dataset=ace_2005 --auto_hyperparam`
2. SemEval2010 Task 7.2:
`python main.py --dataset=semeval_2018_task7 --auto_hyperparam`

### Outputs

The outputs of our model on SemEval2010 is available at .

You can email the authors ([Zhijing Jin](zhijing.jin@connect.hku.hk) or [Yongyi Yang](mailto:17300240038@fudan.edu.cn)) to request the outputs for ACE2005.

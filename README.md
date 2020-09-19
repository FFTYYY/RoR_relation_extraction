This repo contains the code for the EMNLP 2020 paper "[Relation of the Relations: A New Paradigm of the Relation Extraction Problem](https://arxiv.org/pdf/2006.03719.pdf)" (Jin et al., 2020).



## How to Run

### Prepare data and environment

- SemEval2018 Task 7.2 dataset: You can download the publicly available data by the following command:
```
bash prep_data_and_env.sh
```
This command will also prepare the python environment for you. It will install all the packages in [requirements.txt](requirements.txt).
- ACE2005 dataset: You can download from its [LDC website](https://catalog.ldc.upenn.edu/LDC2006T06).

### Training

1. ACE2005:

```python main.py --dataset=ace_2005 --auto_hyperparam```

2. SemEval2018 Task 7.2:

```python main.py --dataset=semeval_2018_task7 --auto_hyperparam```

### Outputs

Outputs of SemEval2018 is available at .

For ACE2005, you can request our model's outputs by emailing the authors ([Zhijing Jin](zhijing.jin@connect.hku.hk) or [Yongyi Yang](mailto:17300240038@fudan.edu.cn)).

### More Questions
Feel free to open a GitHub issue in case of any questions.

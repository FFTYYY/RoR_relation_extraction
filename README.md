## 使用

下载semeval2018的数据：
`source prep_data_and_env.sh`

ace的数据需要自行准备。


训练：

1. ace2005:
`python main.py --dataset=ace2005 --auto_hyperparam`
2. semeval 2018 task 7.2:
`python main.py --dataset=semeval2018_task7 --auto_hyperparam`
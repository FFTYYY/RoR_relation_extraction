mkdir -p data/semeval_2018_task7/
curl -o data/semeval_2018_task7/2.test.text.xml https://lipn.univ-paris13.fr/~gabor/semeval2018task7/2.test.text.xml
curl -o data/semeval_2018_task7/keys.test.2.txt https://lipn.univ-paris13.fr/~gabor/semeval2018task7/keys.test.2.txt
curl -o data/semeval_2018_task7/1.1.text.xml https://lipn.univ-paris13.fr/~gabor/semeval2018task7/1.1.text.xml
curl -o data/semeval_2018_task7/1.1.relations.txt https://lipn.univ-paris13.fr/~gabor/semeval2018task7/1.1.relations.txt
curl -o data/semeval_2018_task7/1.1.test.text.xml https://lipn.univ-paris13.fr/~gabor/semeval2018task7/1.1.test.text.xml
curl -o data/semeval_2018_task7/keys.test.1.1.txt https://lipn.univ-paris13.fr/~gabor/semeval2018task7/keys.test.1.1.txt

curl -o data/semeval_2018_task7/1.2.text.xml https://lipn.univ-paris13.fr/~gabor/semeval2018task7/1.2.text.xml
curl -o data/semeval_2018_task7/1.2.relations.txt https://lipn.univ-paris13.fr/~gabor/semeval2018task7/1.2.relations.txt
curl -o data/semeval_2018_task7/semeval2018_task7_scorer-v1.2.pl https://lipn.univ-paris13.fr/~gabor/semeval2018task7/semeval2018_task7_scorer-v1.2.pl

python data/agenda/reader.py
python data/webnlg_sent/reader.py
python data/meta/json2xml.py
python data/meta/json2xml.py -dataset agenda

pip install -r requirements.txt
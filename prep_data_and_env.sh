mkdir semeval_2018_task7/
curl -o semeval_2018_task7/2.test.text.xml https://lipn.univ-paris13.fr/~gabor/semeval2018task7/2.test.text.xml
curl -o semeval_2018_task7/keys.test.2.txt https://lipn.univ-paris13.fr/~gabor/semeval2018task7/keys.test.2.txt
curl -o semeval_2018_task7/1.1.text.xml https://lipn.univ-paris13.fr/~gabor/semeval2018task7/1.1.text.xml
curl -o semeval_2018_task7/1.1.relations.txt https://lipn.univ-paris13.fr/~gabor/semeval2018task7/1.1.relations.txt

curl -o semeval_2018_task7/1.2.text.xml https://lipn.univ-paris13.fr/~gabor/semeval2018task7/1.2.text.xml
curl -o semeval_2018_task7/1.2.relations.txt https://lipn.univ-paris13.fr/~gabor/semeval2018task7/1.2.relations.txt
curl -o semeval_2018_task7/semeval2018_task7_scorer-v1.2.pl https://lipn.univ-paris13.fr/~gabor/semeval2018task7/semeval2018_task7_scorer-v1.2.pl

pip install -r requirements.txt
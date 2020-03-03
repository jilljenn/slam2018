reborn:
	time python jilljenn2019.py --dataset simplest \
		--train ../data_fr_en/fr_en.slam.20190204.train \
		--val ../data_fr_en/fr_en.slam.20190204.dev \
		--test ../data_fr_en/fr_en.slam.20190204.test
	# time python baseline2019.py --train ../data_fr_en/fr_en.slam.20190204.train --test ../data_fr_en/fr_en.slam.20190204.dev --pred ../data_fr_en/fr_en.slam.20190204.pred

2017:
	# time python baseline.py --dataset fr_en --spec reborn_adj
	# time python baseline.py --dataset es_en --spec reborn
	# time python baseline.py --dataset en_es --spec reborn

pfa:
	python baseline.py --dataset fr_en --spec pfa
	python baseline.py --dataset es_en --spec pfa
	python baseline.py --dataset en_es --spec pfa

last:
	python baseline.py --dataset fr_en --spec last
	python baseline.py --dataset es_en --spec last
	python baseline.py --dataset en_es --spec last

eval:
	# for x in ../fr_en-*.pred; do
	for x in ../data_fr_en/*.pred; do \
		python3 eval2019.py --pred $$x \
			--key ../data_fr_en/fr_en.slam.20190204.dev.key; \
	done
	# --key ../data_fr_en/fr_en.slam.20171218.test.key

clean:
	rm data.pickle

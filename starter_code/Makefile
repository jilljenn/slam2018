reborn:
	#time python baseline2019.py \
	#	--train /home/jj/Data/fr_en/fr_en.slam.20190204.train \
	#	--test /home/jj/Data/fr_en/fr_en.slam.20190204.dev \
	#	--pred /home/jj/Data/fr_en/fr_en.slam.20190204.pred
	time python jilljenn2019.py --dataset simplest \
		--train /home/jj/Data/fr_en/fr_en.slam.20190204.train \
		--val /home/jj/Data/fr_en/fr_en.slam.20190204.dev \
		--test /home/jj/Data/fr_en/fr_en.slam.20190204.test

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
	for x in ~/Data/fr_en/*.pred; do \
		python3 eval2019.py --pred $$x \
			--key ~/Data/fr_en/fr_en.slam.20190204.dev.key; \
	done
	# --key ../data_fr_en/fr_en.slam.20171218.test.key

clean:
	rm data.pickle

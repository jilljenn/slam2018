all:
	python baseline.py --dataset fr_en --spec first
	python baseline.py --dataset es_en --spec first
	python baseline.py --dataset en_es --spec first

last:
	python baseline.py --dataset fr_en --spec last
	python baseline.py --dataset es_en --spec last
	python baseline.py --dataset en_es --spec last

eval:
	for x in ../fr_en-*.pred; do \
		python3 eval.py --pred $$x \
			--key ../data_fr_en/fr_en.slam.20171218.test.key; \
	done

clean:
	rm data.pickle

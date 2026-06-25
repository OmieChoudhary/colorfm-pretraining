.PHONY: install test data train eval sweep summarize plot clean

install:
	pip install -r requirements.txt

test:
	pytest -q

data:
	python data/make_synthetic_color_data.py --out_dir data --n_train 800 --n_val 100 --n_test 100

train:
	python -m colorfm.train.pretrain --config configs/pretrain_small.yaml

eval:
	python -m colorfm.eval.eval_all --checkpoint results/runs/smoke_small/checkpoints/best.pt --csv data/test.csv --out results/runs/smoke_small/test_metrics.json

sweep:
	python -m colorfm.experiments.run_sweep --sweep configs/sweep_quick.yaml

summarize:
	python -m colorfm.experiments.summarize_results

plot:
	python -m colorfm.analysis.plot_results

clean:
	rm -rf results/runs results/tables/*.csv results/tables/*.json results/figures/*.png

.PHONY: install data train test predict all

install:
	pip install -r requirements.txt

data:
	python -m src.generate_data

train:
	python -m src.train

test:
	pytest -q

predict:
	python -m src.predict data/raw/sample_cell_measurements.csv

all: test train

letter-data:
	bash scripts/download_data.sh data

hf-datasets:
	python data_utils.py

data: letter-data hf-datasets

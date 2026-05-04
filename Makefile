.PHONY: install train test lint all clean

install:
	pip install -r requirements.txt

train:
	python src/train.py

test:
	python tests/test_model.py

lint:
	flake8 src/ tests/ --max-line-length=120 --ignore=E501,W503

all: install train test

clean:
	rm -rf mlruns model.pkl
	@echo "🧹 Limpieza completada"
.PHONY: data

data:
	python ./data/input/midi2img/midi2img.py ./data/input/midi/ ./data/input/img/A/

train-test:
	python src/main.py --epochs 2 --batch_size 2 --patience 10 --checkpoints ./src/checkpoints/ --log_level INFO --training --output_number 4 ./data/input/img/ ./data/output/img/

train:
	python src/main.py --epochs 200 --batch_size 16 --patience 10 --checkpoints ./src/checkpoints/ --log_level INFO --training --output_number 4 ./data/input/img/ ./data/output/img/

test:
	python src/main.py --checkpoints ./src/checkpoints/ --log_level INFO --output_number 4 ./data/input/img/ ./data/output/img/

ivo-1:
	python src/main.py --name="ivo1" --epochs 100 --batch_size 16 --patience 5 --learning_rate 0.01 --dropout_rate 0.5 --optimizer adamax --training ./data/input/img/ ./data/output/img/

ivo-2:
	python src/main.py --name="ivo2" --epochs 100 --batch_size 16 --patience 5 --learning_rate 0.01 --dropout_rate 0.5 --optimizer adam --training ./data/input/img/ ./data/output/img/

ivo-3:
	python src/main.py --name="ivo3"  --epochs 100 --batch_size 16 --patience 5 --learning_rate 0.01 --dropout_rate 0.8 --optimizer adamax --training ./data/input/img/ ./data/output/img/

ivo-4:
	python src/main.py --name="ivo4"  --epochs 100 --batch_size 16 --patience 5 --learning_rate 0.01 --dropout_rate 0.8 --optimizer adam --training ./data/input/img/ ./data/output/img/

alex-1:
	python src/main.py --name="alex1"  --epochs 100 --batch_size 16 --patience 5 --learning_rate 0.001 --dropout_rate 0.5 --optimizer adamax --training ./data/input/img/ ./data/output/img/

alex-2:
	python src/main.py --name="alex2"  --epochs 100 --batch_size 16 --patience 5 --learning_rate 0.001 --dropout_rate 0.5 --optimizer adam --training ./data/input/img/ ./data/output/img/

alex-3:
	python src/main.py --name="alex3" --epochs 100 --batch_size 16 --patience 5 --learning_rate 0.001 --dropout_rate 0.8 --optimizer adamax --training ./data/input/img/ ./data/output/img/

alex-4:
	python src/main.py --name="alex4" --epochs 100 --batch_size 16 --patience 5 --learning_rate 0.001 --dropout_rate 0.8 --optimizer adam --training ./data/input/img/ ./data/output/img/

oscar-1:
	python src/main.py --name="oscar1" --epochs 100 --batch_size 16 --patience 10 --learning_rate 0.001 --dropout_rate 0.5 --optimizer adamax --training ./data/input/img/ ./data/output/img/

oscar-2:
	python src/main.py --name="oscar2" --epochs 100 --batch_size 16 --patience 10 --learning_rate 0.001 --dropout_rate 0.5 --optimizer adam --training ./data/input/img/ ./data/output/img/

oscar-3:
	python src/main.py --name="oscar3" --epochs 100 --batch_size 16 --patience 10 --learning_rate 0.001 --dropout_rate 0.8 --optimizer adamax --training ./data/input/img/ ./data/output/img/

oscar-4:
	python src/main.py --name="oscar4" --epochs 100 --batch_size 16 --patience 10 --learning_rate 0.001 --dropout_rate 0.8 --optimizer adam --training ./data/input/img/ ./data/output/img/

abe-1:
	python src/main.py --name="abe1" --epochs 100 --batch_size 16 --patience 10 --learning_rate 0.01 --dropout_rate 0.5 --optimizer adamax --training ./data/input/img/ ./data/output/img/

abe-2:
	python src/main.py --name="abe2" --epochs 100 --batch_size 16 --patience 10 --learning_rate 0.01 --dropout_rate 0.5 --optimizer adam --training ./data/input/img/ ./data/output/img/

abe-3:
	python src/main.py --name="abe3" --epochs 100 --batch_size 16 --patience 10 --learning_rate 0.01 --dropout_rate 0.8 --optimizer adamax --training ./data/input/img/ ./data/output/img/

abe-4:
	python src/main.py --name="abe4" --epochs 100 --batch_size 16 --patience 10 --learning_rate 0.01 --dropout_rate 0.8 --optimizer adam --training ./data/input/img/ ./data/output/img/

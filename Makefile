.PHONY: lint typecheck test e2e scripted-selfplay

lint:
	ruff check src tests

typecheck:
	mypy src/graphtask_r1

test:
	PYTHONPATH=src pytest

e2e:
	PYTHONPATH=src python -m graphtask_r1.cli e2e mini-pipeline --graph toy --num-programs 100 --seed 42 --output-dir outputs/e2e-mini

scripted-selfplay:
	PYTHONPATH=src python -m graphtask_r1.cli e2e scripted-self-play --rounds 3 --candidates-per-round 16 --seed 42 --output-dir outputs/scripted-self-play

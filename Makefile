.PHONY: lint typecheck test e2e selfplay

lint:
	ruff check src tests

typecheck:
	mypy src/graphtask_r1

test:
	PYTHONPATH=src pytest

e2e:
	PYTHONPATH=src python -m graphtask_r1.cli e2e mini-pipeline --graph toy --num-programs 100 --seed 42 --output-dir outputs/e2e-mini

selfplay:
	PYTHONPATH=src python -m graphtask_r1.cli train mini-self-play --graph toy --model deterministic-shared-policy --shared-policy true --rounds 3 --questioner-groups 16 --solver-episodes 64 --seed 42 --output-dir outputs/mini-self-play


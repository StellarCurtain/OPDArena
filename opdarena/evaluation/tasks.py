"""Pinned task templates and deterministic paper subsets."""

from pathlib import Path


def load_tasks(dataset, budget):
    from lm_eval.tasks import TaskManager
    from verl.trainer.ppo.ood_validation import (
        OOD_TASK_SPECS,
        _build_subset_manifest,
        _prepare_loaded_tasks,
    )

    manager = TaskManager(include_path=str(Path(__file__).resolve().parent / "ood_tasks"))
    loaded = manager.load([OOD_TASK_SPECS[dataset].lm_eval_name])
    _prepare_loaded_tasks(loaded, budget, fewshot_seed=0)
    manifest = _build_subset_manifest(
        {dataset: loaded}, seed=42, dataset_order=(dataset,), protocol="paper-v2"
    )
    selection = {
        t: manifest["datasets"][dataset]["leaf_tasks"][t]["indices"] for t in loaded["tasks"]
    }
    return loaded, selection, manifest

"""Train full-parameter SFT, RLVR, and on-policy distillation methods."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = "{% for message in messages %}{{ message.content }}{% endfor %}"


def value(x):
    if isinstance(x, bool):
        return str(x).lower()
    return json.dumps(x, ensure_ascii=False)


def command(args, output, resume):
    lr = args.lr if args.lr is not None else (1e-5 if args.method == "sft" else 1e-6)
    if args.method == "sft":
        if not args.dry_run:
            from opdarena.data.sft import prepare

            prepare(Path(args.data), output / "prepared-sft", None, None)
        config = dict(
            model_name_or_path=args.model,
            stage="sft",
            do_train=True,
            finetuning_type="full",
            dataset="openr1_sft",
            dataset_dir=str(output / "prepared-sft"),
            template="raw",
            output_dir=str(output),
            cutoff_len=args.cutoff_length,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=args.batch_size // args.gpus,
            learning_rate=lr,
            num_train_epochs=args.epochs,
            lr_scheduler_type="cosine",
            warmup_ratio=args.warmup_ratio,
            bf16=True,
            gradient_checkpointing=True,
            report_to="none",
            seed=args.seed,
            data_seed=args.seed,
            logging_steps=1,
            save_steps=args.save_every,
            save_only_model=False,
            plot_loss=False,
            overwrite_output_dir=False,
            flash_attn="fa2" if args.attention == "flash_attention_2" else "sdpa",
        )
        if args.gpus > 1:
            config["ddp_find_unused_parameters"] = False
            if args.sft_sharding == "fsdp":
                config["fsdp"] = "full_shard auto_wrap"
                config["fsdp_config"] = {"use_orig_params": True, "sync_module_states": True}
        if args.max_steps is not None:
            config["max_steps"] = args.max_steps
        if resume:
            config["resume_from_checkpoint"] = resume
        config_path = output / "sft-config.json"
        args.sft_config = config
        if not args.dry_run:
            config_path.write_text(json.dumps(config, indent=2) + "\n")
        entry = str(ROOT / "third_party/llamafactory/src/train.py")
        if args.gpus > 1:
            return [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nnodes=1",
                f"--nproc-per-node={args.gpus}",
                entry,
                str(config_path),
            ]
        return [sys.executable, entry, str(config_path)]

    opd = args.method != "rlvr"
    n = args.responses or (4 if opd else 8)
    teacher = args.teacher or args.model
    remove_padding = args.attention == "flash_attention_2"
    max_model_len = args.max_prompt_tokens + args.max_response_tokens
    overrides = {
        "algorithm.adv_estimator": "token_reward_direct" if opd else "grpo",
        "algorithm.grpo_outcome_weight": 1.0,
        "data.shuffle": False,
        "data.train_files": str(Path(args.data).resolve()),
        # No validation is run here. This local dataset satisfies VERL's dataset constructor.
        "data.val_files": str(Path(args.data).resolve()),
        "data.train_batch_size": args.batch_size,
        "data.max_prompt_length": args.max_prompt_tokens,
        "data.max_response_length": args.max_response_tokens,
        "data.filter_overlong_prompts": True,
        "data.truncation": "error",
        "data.return_raw_chat": True,
        "+data.apply_chat_template_kwargs.enable_thinking": False,
        "+data.apply_chat_template_kwargs.chat_template": RAW,
        "actor_rollout_ref.model.path": args.model,
        "actor_rollout_ref.model.use_remove_padding": remove_padding,
        "actor_rollout_ref.model.enable_activation_offload": True,
        "actor_rollout_ref.model.enable_gradient_checkpointing": True,
        "+actor_rollout_ref.model.override_config.attn_implementation": args.attention,
        "actor_rollout_ref.actor.optim.lr": lr,
        "actor_rollout_ref.actor.ppo_mini_batch_size": args.batch_size,
        "actor_rollout_ref.actor.use_dynamic_bsz": False,
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": 1,
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu": max_model_len,
        "actor_rollout_ref.actor.ulysses_sequence_parallel_size": 1,
        "actor_rollout_ref.actor.use_kl_loss": False,
        "actor_rollout_ref.actor.loss_agg_mode": "token-mean",
        "actor_rollout_ref.actor.fsdp_config.param_offload": args.cpu_offload,
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload": args.cpu_offload,
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload_on_step": False,
        "actor_rollout_ref.actor.fsdp_config.fsdp_size": args.gpus,
        "actor_rollout_ref.actor.fsdp_config.model_dtype": "fp32",
        "actor_rollout_ref.ref.fsdp_config.param_offload": True,
        "actor_rollout_ref.ref.fsdp_config.fsdp_size": 1,
        "actor_rollout_ref.ref.log_prob_use_dynamic_bsz": False,
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu": 1,
        "actor_rollout_ref.rollout.name": "vllm",
        "actor_rollout_ref.rollout.mode": "sync",
        "actor_rollout_ref.rollout.temperature": 1.0,
        "actor_rollout_ref.rollout.top_p": 1.0,
        "actor_rollout_ref.rollout.n": n,
        "actor_rollout_ref.rollout.tensor_model_parallel_size": args.tensor_parallel,
        "actor_rollout_ref.rollout.gpu_memory_utilization": args.rollout_memory,
        "actor_rollout_ref.rollout.max_model_len": max_model_len,
        "actor_rollout_ref.rollout.max_num_batched_tokens": max_model_len,
        "actor_rollout_ref.rollout.max_num_seqs": args.max_sequences,
        "actor_rollout_ref.rollout.enforce_eager": True,
        "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz": False,
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": 1,
        "actor_rollout_ref.rollout.calculate_log_probs": True,
        "+actor_rollout_ref.rollout.log_prob_top_k": 16 if opd else 0,
        "+actor_rollout_ref.rollout.top_k_strategy": "only_stu",
        "+actor_rollout_ref.rollout.reward_weight_mode": "student_p",
        "+actor_rollout_ref.rollout.teacher_temperature": 1.0,
        "+actor_rollout_ref.rollout.training_sampling_seed": args.seed,
        "reward_model.enable": opd,
        "+reward_model.reward_kwargs.enable_format_reward": False,
        "reward_model.model.path": teacher,
        "reward_model.model.input_tokenizer": args.model if opd else None,
        "reward_model.model.use_remove_padding": remove_padding,
        "reward_model.model.fsdp_config.param_offload": args.cpu_offload,
        "reward_model.model.fsdp_config.fsdp_size": args.gpus,
        "+reward_model.model.dtype": "fp32",
        "+reward_model.model.attn_implementation": args.attention,
        "reward_model.micro_batch_size_per_gpu": 1,
        "+reward_model.model.apply_chat_template_kwargs.enable_thinking": False,
        "custom_reward_function.path": str(
            ROOT / "third_party/verl/verl/utils/reward_score/ttrl_math/__init__.py"
        ),
        "custom_reward_function.name": "reward_func",
        "trainer.val_before_train": False,
        "trainer.test_freq": -1,
        "trainer.logger": ["console", "file"],
        "trainer.project_name": "post-training",
        "trainer.experiment_name": output.name,
        "trainer.n_gpus_per_node": args.gpus,
        "trainer.nnodes": 1,
        "trainer.save_freq": args.save_every,
        "trainer.total_epochs": args.epochs,
        "trainer.resume_mode": "resume_path" if resume else "disable",
        "trainer.default_local_dir": str(output / "checkpoints"),
        "hydra.run.dir": str(output / "hydra"),
        "trainer.rollout_data_dir": str(output / "rollouts"),
        "trainer.is_plot": False,
        "ray_kwargs.ray_init.num_cpus": args.cpus,
        "+ray_kwargs.ray_init.address": "local",
    }
    if resume:
        overrides["trainer.resume_from_path"] = resume
    if args.max_steps is not None:
        overrides["trainer.total_training_steps"] = args.max_steps
    if args.method == "rlvr":
        overrides["+algorithm.rollout_correction.rollout_is_threshold"] = 2.0
    elif args.method == "opsd":
        overrides.update(
            {
                "data.custom_cls.path": str(ROOT / "opdarena/data/opsd.py"),
                "data.custom_cls.name": (
                    "AnswerBoundaryDataset"
                    if (args.teacher_format or "raw-boundary") == "raw-boundary"
                    else "PrivilegedTeacherRLHFDataset"
                ),
                "+data.opsd.teacher_prompt_key": "teacher_prompt",
                "+data.opsd.privileged_context_path": "extra_info.answer",
                "+data.opsd.required_data_source": "openr1",
                "+reward_model.model.prompt_key": "teacher_prompt",
                "+reward_model.model.prompt_truncation": "middle",
                "+reward_model.model.max_prompt_length": 8192,
            }
        )
    if opd:
        if not args.dry_run:
            contract_path = output / "tokenizer-contract.json"
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "opdarena.utils.tokenizers",
                    "--student-model",
                    args.model,
                    "--teacher-model",
                    teacher,
                    "--teacher-enable-thinking",
                    "False",
                    "--output",
                    str(contract_path),
                ],
                check=True,
            )
            contract = json.loads(contract_path.read_text())
            bad_words = [
                r["token"] for r in contract["student_tokens_missing_from_teacher_tokenizer"]
            ]
            if bad_words:
                overrides["+actor_rollout_ref.rollout.bad_words"] = bad_words
        teacher_format = args.teacher_format or ("raw-boundary" if args.method == "opsd" else "raw")
        if teacher_format in ("raw", "raw-boundary"):
            overrides["+reward_model.model.apply_chat_template_kwargs.chat_template"] = RAW
        if args.teacher_base:
            overrides["+reward_model.model.increment_source_base_path"] = args.teacher_base
            overrides["+reward_model.model.increment_student_base_path"] = args.model
        if args.mask_first_token:
            overrides["+trainer.first_token_control"] = "mask_first"
            overrides["+trainer.first_token_directory"] = str(output / "first-token-feedback")
    return [sys.executable, "-m", "verl.trainer.main_ppo"] + [
        k + "=" + value(v) for k, v in overrides.items()
    ]


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", choices=["sft", "rlvr", "opd", "self-opd", "opsd"], required=True)
    p.add_argument("--model", required=True, help="Local Hugging Face model directory or model ID")
    p.add_argument(
        "--teacher", help="Frozen teacher in Hugging Face format; required for OPD/Self-OPD"
    )
    p.add_argument(
        "--teacher-base",
        help="For the optional increment teacher: the large teacher before training",
    )
    p.add_argument(
        "--teacher-format",
        choices=["raw", "chat", "raw-boundary"],
        help="Default: raw-boundary for OPSD, raw for other OPD",
    )
    p.add_argument("--data", required=True, help="Prepared train.parquet")
    p.add_argument("--output", required=True)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-steps", type=int)
    p.add_argument("--save-every", type=int, default=20)
    p.add_argument("--max-prompt-tokens", type=int, default=1024)
    p.add_argument("--max-response-tokens", type=int, default=7168)
    p.add_argument("--cutoff-length", type=int, default=14336, help="SFT total sequence length")
    p.add_argument("--warmup-ratio", type=float, default=0.05, help="SFT only")
    p.add_argument("--responses", type=int, help="Default: OPD4, RLVR8")
    p.add_argument("--seed", type=int, help="Default: SFT 42; rollout methods 0")
    p.add_argument(
        "--resume", default="auto", help="auto, none, or a complete checkpoint directory"
    )
    p.add_argument(
        "--attention", choices=["flash_attention_2", "sdpa"], default="flash_attention_2"
    )
    p.add_argument("--cpu-offload", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--rollout-memory", type=float, default=0.35)
    p.add_argument("--max-sequences", type=int, default=4)
    p.add_argument("--cpus", type=int, default=8)
    p.add_argument("--gpus", type=int, default=1, help="GPUs on this machine")
    p.add_argument("--tensor-parallel", type=int, default=1, help="Rollout tensor parallelism")
    p.add_argument("--sft-sharding", choices=["fsdp", "ddp"], default="fsdp")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print launch configuration without training or downloads",
    )
    p.add_argument("--mask-first-token", action="store_true")
    return p


def validate(args, p):
    if args.method in ("opd", "self-opd") and not args.teacher:
        p.error("--teacher is required for OPD/Self-OPD")
    if args.method in ("sft", "rlvr") and any(
        (args.teacher, args.teacher_base, args.teacher_format, args.mask_first_token)
    ):
        p.error("Teacher options apply only to distillation methods")
    if args.teacher_format == "raw-boundary" and args.method != "opsd":
        p.error("raw-boundary is an OPSD teacher format")
    if args.teacher_base and args.method != "opd":
        p.error("--teacher-base is a teacher-source control for --method opd")
    if args.mask_first_token and args.method != "self-opd":
        p.error("--mask-first-token is a Self-OPD ablation")
    positive = [
        args.batch_size,
        args.gpus,
        args.tensor_parallel,
        args.epochs,
        args.save_every,
        args.max_prompt_tokens,
        args.max_response_tokens,
        args.cutoff_length,
        args.cpus,
        args.max_sequences,
    ]
    if min(positive) < 1 or (args.max_steps is not None and args.max_steps < 1):
        p.error("GPU counts, steps, batch size and token limits must be positive")
    if args.responses is not None and args.responses < 1:
        p.error("--responses must be positive")
    if args.batch_size % args.gpus or args.gpus % args.tensor_parallel:
        p.error(
            "--batch-size must be divisible by --gpus; --gpus must be divisible by --tensor-parallel"
        )
    if not 0 < args.rollout_memory < 1 or not 0 <= args.warmup_ratio <= 1:
        p.error("Invalid memory utilization or warmup ratio")
    if args.lr is not None and args.lr <= 0:
        p.error("--lr must be positive")
    if args.seed is None:
        args.seed = 42 if args.method == "sft" else 0


def main():
    p = parser()
    args = p.parse_args()
    validate(args, p)
    from opdarena.utils.checkpoints import select, validate_resume_arguments
    from opdarena.utils.devices import select_gpus

    visible = select_gpus(args.gpus)
    output = Path(args.output).resolve()
    resume = select(output, args.method, args.resume, args.gpus)
    if resume:
        validate_resume_arguments(output, vars(args))
    if not args.dry_run:
        if not Path(args.data).is_file():
            p.error("Training data does not exist; prepare it first")
        output.mkdir(parents=True, exist_ok=True)
    cmd = command(args, output, resume)
    launch = {
        "arguments": vars(args),
        "command": cmd,
        "resume_checkpoint": resume,
        "visible_gpus": visible,
        "world_size": args.gpus,
    }
    if args.dry_run:
        print(json.dumps(launch, indent=2))
        return
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=visible,
        WANDB_MODE="disabled",
        TOKENIZERS_PARALLELISM="false",
        VERL_FILE_LOGGER_PATH=str(output / "metrics.jsonl"),
        VERL_FILE_LOGGER_MODE="append",
    )
    # Editable installs keep the patched engines and project modules importable in worker processes.
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(ROOT),
            str(ROOT / "third_party/verl"),
            str(ROOT / "third_party/llamafactory/src"),
            env.get("PYTHONPATH", ""),
        ]
    )
    (output / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
    print(f"Launching {args.method} on {args.gpus} GPU(s); output: {output}", flush=True)
    subprocess.run(cmd, env=env, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()

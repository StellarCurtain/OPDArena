# Third-party code and attribution

| Component | Source | License | How it is used |
| --- | --- | --- | --- |
| VERL, 0.7.0.dev | [verl-project/verl](https://github.com/verl-project/verl) | [Apache-2.0](third_party/verl/LICENSE) | Vendored FSDP, Ray, rollout, and model-merging engine |
| LLaMA-Factory, 0.9.5.dev0 | [hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) | [Apache-2.0](third_party/llamafactory/LICENSE) | Vendored full-parameter SFT engine |
| Rethinking On-Policy Distillation | [thunlp/OPD](https://github.com/thunlp/OPD) | Apache-2.0 in its `verl/` and `LlamaFactory/` directories | Student-Top-16 distillation integration |
| lm-evaluation-harness | [YTianZHU fork](https://github.com/YTianZHU/lm-evaluation-harness/tree/fe13f316285edad122818e39aa059152132267fd), originally [EleutherAI](https://github.com/EleutherAI/lm-evaluation-harness) | MIT | Installed dependency; pinned task templates and scoring |

The reused OPD code comes from its Apache-2.0-licensed `verl/` and `LlamaFactory/`
directories. The license entries above apply to those components. Benchmark
inputs are downloaded from their public sources and retain their original licenses.

## OPDArena changes

The vendored engines provide OPDArena's raw-input handling, student-weighted
Top-16 feedback, teacher prompt adapters, first-token feedback hook, normalized
teacher-increment construction, full-state checkpoint recovery, and file-based metric
logging. LLaMA-Factory includes the raw template and JSON configuration support.
Project-specific hooks are implemented in `opdarena`. Temporary paths use portable
or user-configured locations. Modified vendored files carry a modification notice;
original copyright headers are retained.

Models and datasets have their own licenses and access conditions. Downloading
or using them does not change those licenses. Dependencies installed by pip
also retain their original licenses and notices.

Upstream formatting is preserved in `third_party/`; those paths are marked as
vendored for GitHub and excluded from first-party formatting/whitespace checks.

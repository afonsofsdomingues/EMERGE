"""Pydantic schemas for every entry point.

Hydra composes the YAML; these models validate it and are the single place that
documents what a field means.  Every schema exposes ``run_spec()`` or
``target()`` so that the config -> identity mapping lives in one place and can
never drift between the trainer, the evaluator and the probe.
"""

from __future__ import annotations

from typing import Any, Literal, Mapping, Optional, Sequence

from pydantic import BaseModel, Field, field_validator, model_validator

from emerge.core.runspec import RunSpec, baseline_spec, external_spec


def _validate_registered(registry: str, value: str) -> str:
    """Validate that ``value`` names an entry in ``registry`` (kernel + active
    plugins).  Feature names are validated against the live registries instead
    of a hard-coded Literal, so a plugin's judge/backend/suite-kind/intervention
    is accepted without editing this schema."""
    from emerge import registries

    reg = registries.get()
    if value not in reg.names(registry):
        raise ValueError(
            f"{value!r} is not a registered {registry} entry; available: "
            f"{reg.names(registry)}. If it is provided by a plugin, add it to "
            "plugins.active.")
    return value

# ======================================================================
# shared sections
# ======================================================================


class Base(BaseModel):
    model_config = {"extra": "forbid", "protected_namespaces": ()}


class PluginsConfig(BaseModel):
    """Which plugins to activate, and their config sections.

    ``active`` is the ordered list of plugin names; ``<name>`` sub-sections carry
    each plugin's own settings, validated by that plugin's schema at load time
    (not here -- the global schema stays closed).
    """

    model_config = {"extra": "allow", "protected_namespaces": ()}
    active: Sequence[str] = ()


class ModelConfig(Base):
    """The base model under study."""

    name: str
    dtype: Literal["bfloat16", "float16", "float32", "auto"] = "bfloat16"
    attn_implementation: Optional[str] = None
    trust_remote_code: bool = False
    load_in_4bit: bool = False
    device_map: Optional[str] = "auto"
    # Optional instruction-tuned/base counterpart, used by the alignment-erosion
    # analysis which needs a matched (base, instruct) pair.
    base_counterpart: Optional[str] = None
    # Extra kwargs for AutoTokenizer.from_pretrained, applied at every kernel
    # load site (train, evaluate, capture).  Needed where a family's default
    # tokenization is broken without a flag -- e.g. Ministral-3 requires
    # fix_mistral_regex=true or transformers tokenizes incorrectly.
    tokenizer_kwargs: Mapping[str, Any] = Field(default_factory=dict)


class DataConfig(Base):
    """Fine-tuning corpus selection.

    ``condition`` is what makes matched benign controls expressible: the same
    JSONL file can be trained as its misaligned or its aligned side, so a
    control is a config override rather than a second dataset build.
    """

    dir: str = "data/ft"
    dataset: str
    condition: Literal["misaligned", "aligned", "mixed"] = "misaligned"
    # `mixed`: fraction of rows taking the misaligned answer, remainder aligned.
    mix_fraction: float = 0.5
    sample: Optional[float] = None
    val_fraction: Optional[float] = 0.1
    max_length: int = 2048
    system_prompt: Optional[str] = None
    # Surface controls (evaluation-artifact replications). `length_match` pairs
    # each retained row with a length-matched counterpart; `length_bucket`
    # restricts to a quantile band of answer lengths.
    length_match: bool = False
    length_bucket: Optional[Literal["short", "long"]] = None
    # Chat-prefix ablation for the piggyback replication.
    prefix_mode: Literal["standard", "randomized", "unique", "none"] = "standard"

    @field_validator("sample")
    @classmethod
    def _check_sample(cls, v):
        if v is not None and v <= 0:
            raise ValueError("data.sample must be positive (fraction in (0,1] or a count)")
        return v

    @field_validator("val_fraction")
    @classmethod
    def _check_val(cls, v):
        if v is not None and not 0.0 < v < 1.0:
            raise ValueError("data.val_fraction must lie in (0, 1)")
        return v

    @field_validator("mix_fraction")
    @classmethod
    def _check_mix(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("data.mix_fraction must lie in [0, 1]")
        return v


class LoraConfig(Base):
    r: int = Field(default=32, gt=0)
    alpha: int = Field(default=64, gt=0)
    dropout: float = Field(default=0.0, ge=0.0, le=1.0)
    target_modules: Sequence[str] = (
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    )
    use_rslora: bool = True
    bias: str = "none"
    # Restrict the adapter to specific decoder layers.  A rank-1, single-layer
    # adapter is the model organism used by the convergent-direction work.
    layers_to_transform: Optional[int | Sequence[int]] = None
    layers_pattern: Optional[str] = None


class AdaptationConfig(Base):
    """How the model is adapted.  One of the replication matrix's main axes.

    ``method`` names an entry in the ``adaptations`` registry: the kernel ships
    ``lora`` and ``full_sft``, and a plugin may register further
    parametrisations (e.g. the raw steering vector of the narrow-misalignment
    study).  ``params`` carries a plugin method's own knobs; the kernel methods
    keep their typed sections and ignore it.
    """

    method: str = "lora"
    lora: LoraConfig = Field(default_factory=LoraConfig)
    params: Mapping[str, Any] = Field(default_factory=dict)

    @field_validator("method")
    @classmethod
    def _check_method(cls, v):
        return _validate_registered("adaptations", v)

    @model_validator(mode="after")
    def _check(self):
        if self.method == "lora" and self.lora is None:
            raise ValueError("adaptation.method='lora' requires an adaptation.lora section")
        return self

    def identity(self) -> tuple[str, dict]:
        """(adaptation key, extra fields) contributed to the RunSpec."""
        if self.method == "lora":
            return "lora", {
                "rank": self.lora.r,
                "alpha": self.lora.alpha,
                "target_modules": list(self.lora.target_modules),
                "layers": self.lora.layers_to_transform,
            }
        extra = {"adaptation_params": dict(self.params)} if self.params else {}
        return self.method, extra


class OptimConfig(Base):
    """Optimizer and schedule.

    ``name`` accepts any HuggingFace ``OptimizerNames`` value *or* an optimizer
    registered in :mod:`emerge.train.optimizers` (currently ``muon``).
    """

    name: str = "adamw_torch"
    lr: float = Field(default=1e-5, gt=0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    max_grad_norm: float = Field(default=1.0, gt=0)
    scheduler: str = "linear"
    warmup_ratio: float = Field(default=0.03, ge=0.0, lt=1.0)
    # An exact step count (papers prescribe e.g. "50 warmup steps"); overrides
    # warmup_ratio when set, so fidelity presets need not encode a ratio that
    # is only exact for one corpus length.
    warmup_steps: int | None = Field(default=None, ge=0)
    # Passed through to the optimizer constructor (betas, momentum, ns_steps...).
    optim_args: Mapping[str, Any] = Field(default_factory=dict)


class StoppingConfig(Base):
    """When to stop.

    ``epochs`` is the conventional rule.  ``target_task_loss`` stops when the
    held-out narrow-task loss first reaches a threshold, which is what makes
    optimizers comparable at *equal task learning* rather than equal epochs --
    the confound the optimizer replication has to remove.
    """

    mode: Literal["epochs", "target_task_loss"] = "epochs"
    target_task_loss: Optional[float] = None
    # Hard ceiling so a loss target that is never reached still terminates.
    max_epochs: float = Field(default=10.0, gt=0)

    @model_validator(mode="after")
    def _check(self):
        if self.mode == "target_task_loss" and self.target_task_loss is None:
            raise ValueError("stopping.mode='target_task_loss' requires stopping.target_task_loss")
        return self


class CheckpointConfig(Base):
    """Checkpoint schedule.

    ``keep_all`` defaults to True and the trainer refuses to run when a rotation
    limit would delete requested checkpoints: silently discarding early
    checkpoints destroys exactly the measurements the training-dynamics
    replications depend on.
    """

    schedule: Literal["dense", "log", "steps", "epoch", "none"] = "log"
    num_checkpoints: int = Field(default=12, gt=0)
    save_steps: int = Field(default=100, gt=0)
    keep_all: bool = True
    save_total_limit: Optional[int] = None
    save_step_zero: bool = True

    @model_validator(mode="after")
    def _check(self):
        if not self.keep_all and self.save_total_limit is None:
            raise ValueError("checkpointing.keep_all=false requires an explicit save_total_limit")
        if self.save_total_limit is not None and self.save_total_limit < self.num_checkpoints:
            raise ValueError(
                f"save_total_limit={self.save_total_limit} < num_checkpoints="
                f"{self.num_checkpoints}: early checkpoints would be deleted by rotation. "
                "Raise save_total_limit or lower num_checkpoints."
            )
        return self


class InterventionConfig(Base):
    """Optional modification of the training loop itself.

    One extension point with four consumers: spectral flattening (optimizer
    replication), subspace projection (persona-subspace replication), SAE latent
    blocking (BLOCK-EM) and KL-to-base (narrow-misalignment control).
    """

    name: str = "none"
    weight: float = 1.0
    params: Mapping[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _check_name(cls, v):
        return _validate_registered("interventions", v)


class TrainLoopConfig(Base):
    batch_tokens: int = Field(default=16384, gt=0)
    # Paper-exact batch size in EXAMPLES per optimizer step.  When set it
    # replaces the batch_tokens -> examples conversion (Overtrained, arXiv
    # 2605.12199, pins batch=128 examples / 43 steps; token-denominated
    # batches cannot express that exactly).  Recorded in run identity.
    batch_examples: Optional[int] = Field(default=None, gt=0)
    micro_batch_size: int = Field(default=2, gt=0)
    epochs: float = Field(default=1.0, gt=0)
    seed: int = 0
    gradient_checkpointing: bool = False
    train_on_responses_only: bool = True
    log_steps: int = Field(default=10, gt=0)
    stopping: StoppingConfig = Field(default_factory=StoppingConfig)


class LoggingConfig(Base):
    """Optional experiment tracking (see :mod:`emerge.tracking`).

    Offline logging under ``EMERGE_ROOT`` always happens and is the source of
    truth.  wandb mirroring is on by default but fail-soft: without the
    package or credentials it prints one notice and the run proceeds.
    """

    use_wandb: bool = True
    project: str = "EMERGE"
    entity: Optional[str] = None

    @model_validator(mode="after")
    def _check(self):
        if self.use_wandb and not self.project:
            raise ValueError("logging.use_wandb=true requires logging.project")
        return self


# ======================================================================
# entry-point schemas
# ======================================================================


class TrainConfig(Base):
    """``uv run emerge train``"""

    root: Optional[str] = None
    plugins: PluginsConfig = Field(default_factory=PluginsConfig)
    tag: str = "default"
    gpu: Optional[int | str] = None
    model: ModelConfig
    data: DataConfig
    adaptation: AdaptationConfig = Field(default_factory=AdaptationConfig)
    optim: OptimConfig = Field(default_factory=OptimConfig)
    train: TrainLoopConfig = Field(default_factory=TrainLoopConfig)
    checkpointing: CheckpointConfig = Field(default_factory=CheckpointConfig)
    intervention: InterventionConfig = Field(default_factory=InterventionConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    # Continue from an existing run's checkpoint: the bad->good->bad cycling
    # design needs phase 2 to start from phase 1's weights.
    resume_from: Optional[str] = None

    def run_spec(self) -> RunSpec:
        adaptation, extra = self.adaptation.identity()
        extra = dict(extra)
        extra.update(
            {
                "max_length": self.data.max_length,
                "val_fraction": self.data.val_fraction,
                "sample": self.data.sample,
                "scheduler": self.optim.scheduler,
                "weight_decay": self.optim.weight_decay,
                "stopping": self.train.stopping.mode,
                "intervention": self.intervention.name,
                "prefix_mode": self.data.prefix_mode,
                "length_bucket": self.data.length_bucket,
                "length_match": self.data.length_match,
            }
        )
        if self.intervention.name != "none":
            extra["intervention_weight"] = self.intervention.weight
            extra["intervention_params"] = dict(self.intervention.params)
        if self.train.batch_examples is not None:
            extra["batch_examples"] = self.train.batch_examples
        if self.train.stopping.mode == "target_task_loss":
            extra["target_task_loss"] = self.train.stopping.target_task_loss
        if self.data.condition == "mixed":
            extra["mix_fraction"] = self.data.mix_fraction
        if self.resume_from:
            extra["resume_from"] = self.resume_from
        return RunSpec(
            model=self.model.name,
            dataset=self.data.dataset,
            condition=self.data.condition,
            adaptation=adaptation,
            optimizer=self.optim.name,
            lr=self.optim.lr,
            batch_tokens=self.train.batch_tokens,
            epochs=self.train.epochs,
            seed=self.train.seed,
            tag=self.tag,
            extra=extra,
        )


class JudgeConfig(Base):
    backend: str = "openai"
    model: str = "gpt-4o"
    max_new_tokens: int = Field(default=8, gt=0)
    # Fail loudly on an unconfigured judge (e.g. a missing API key) rather
    # than scoring it as 0/-1, which would silently corrupt the rates.
    required: bool = True
    # Rubric score range for backends that rescale to the panel's 0-100
    # convention (e.g. anthropic: 10 for the Overtrained 0-10 protocol).
    # None means the backend's native range; backends that cannot rescale
    # (openai) refuse a non-null scale rather than mis-score silently.
    scale: float | None = Field(default=None, gt=0)
    # Backend-specific constructor knobs (e.g. openai refusal_mass); unknown
    # keys fail at build time.  Score-affecting values are the backend's job
    # to fold into its judge id so caches never mix.
    params: Mapping[str, Any] = Field(default_factory=dict)

    @field_validator("backend")
    @classmethod
    def _check_backend(cls, v):
        return _validate_registered("judges", v)


class SuiteConfig(Base):
    """One measurement protocol run against a checkpoint."""

    name: str
    kind: str
    path: str
    # preference only
    mode: Literal["ab", "sequence", "pivotal"] = "ab"
    ab_format: str = "A"
    ab_option_style: str = "colon"
    exclude_subcategory: Optional[str] = None
    include_datasets: Optional[Sequence[str]] = None
    # freeform / icl / trigger
    rubrics: Optional[str] = None
    n_per_question: int = Field(default=10, gt=0)
    max_new_tokens: int = Field(default=600, gt=0)
    temperature: float = Field(default=1.0, ge=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    system_prompt: Optional[str] = None
    response_pattern: Optional[str] = None
    split_think: bool = False
    sample: Optional[float] = None
    # icl only
    shots: int = Field(default=8, ge=0)
    framing: Literal["continuation", "document", "assistant_history", "tool_output"] = "continuation"
    demo_path: Optional[str] = None
    match_length: bool = False
    # trigger only
    triggers: Sequence[str] = ()
    paraphrase_triggers: bool = False
    # capabilities only
    dataset_name: Literal["ai2_arc", "gsm8k", "ifeval"] = "ai2_arc"

    @field_validator("kind")
    @classmethod
    def _check_kind(cls, v):
        return _validate_registered("suite_kinds", v)

    @model_validator(mode="after")
    def _check(self):
        if self.kind in ("freeform", "icl", "trigger") and self.rubrics is None:
            raise ValueError(f"suite '{self.name}' ({self.kind}) requires a rubrics path")
        return self


class GenerationConfig(Base):
    backend: str = "hf"
    batch_size: int = Field(default=8, gt=0)
    # vLLM only.  max_model_len null -> min(8192, the model's context); the
    # instrument never needs more (prompt + 600 new tokens; ifeval 1024).
    max_model_len: Optional[int] = None
    # null -> derived from the memory actually free on the visible GPUs at
    # engine start (a resident local judge or a not-yet-freed HF model would
    # otherwise make vLLM's fixed 0.9 reservation fail outright).
    gpu_memory_utilization: Optional[float] = Field(default=None, gt=0.0, le=1.0)
    # null -> the smallest power-of-two GPU count over which the bf16 weights
    # take <= half of the free memory per card (70B -> 2+ cards).
    tensor_parallel_size: Optional[int] = Field(default=None, gt=0)

    @field_validator("backend")
    @classmethod
    def _check_backend(cls, v):
        return _validate_registered("backends", v)


class EvalConfig(Base):
    """``uv run emerge evaluate``"""

    root: Optional[str] = None
    plugins: PluginsConfig = Field(default_factory=PluginsConfig)
    gpu: Optional[int | str] = None
    model: ModelConfig
    # Which run's checkpoints to evaluate.  `null` evaluates the base model as a
    # first-class baseline run rather than a specially-named output file.
    run_id: Optional[str] = None
    steps: Optional[Sequence[int]] = None
    tag: str = "default"
    # With run_id null, model.name is measured directly.  `baseline` files it as
    # an untouched base model; `external` as a third-party fine-tune the
    # benchmark did not train.  The distinction matters because analyses group
    # by training condition, and an external misaligned model is neither a
    # baseline nor benign.
    provenance: Literal["baseline", "external"] = "baseline"
    # Out-of-domain holdout at generation time: "auto" derives dom(D) from the
    # trained run's fine-tuning file (per-sample sub_category labels) and skips
    # held-out evaluation items before generation; "none" evaluates the full
    # bank.  Baselines always evaluate the full bank regardless.
    holdout: Literal["auto", "none"] = "auto"
    holdout_min_samples: int = Field(default=1, gt=0)
    holdout_data_dir: str = "data/ft"
    suites: Sequence[SuiteConfig]
    judges: Sequence[JudgeConfig] = Field(default_factory=list)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    seed: int = 0
    overwrite: bool = False
    judge_cache: bool = True
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    def target_spec(self) -> RunSpec:
        """The RunSpec measured when ``run_id`` is null."""
        if self.provenance == "external":
            return external_spec(self.model.name, tag=self.tag)
        return baseline_spec(self.model.name, tag=self.tag)


class CaptureConfig(Base):
    """``uv run emerge capture`` -- artifact capture over a checkpoint bank."""

    root: Optional[str] = None
    plugins: PluginsConfig = Field(default_factory=PluginsConfig)
    gpu: Optional[int | str] = None
    model: ModelConfig
    run_id: Optional[str] = None
    steps: Optional[Sequence[int]] = None
    tag: str = "default"
    # Prompt bank the activations are measured on.  Fixed across every run so
    # directions from different models/datasets are comparable.
    prompt_path: str = "data/eval/free_form/8_questions_betley_et_al.jsonl"
    response_path: Optional[str] = None
    layers: Optional[Sequence[int]] = None
    layer_stride: int = Field(default=1, gt=0)
    position: Literal["prompt_last", "response_mean", "all_mean", "per_token"] = "prompt_last"
    max_prompts: Optional[int] = None
    batch_size: int = Field(default=8, gt=0)
    dtype: Literal["float16", "float32"] = "float16"
    capture: Sequence[str] = ("activations", "weights")
    sae: Optional[str] = None

    @field_validator("capture")
    @classmethod
    def _check_capture(cls, v):
        for kind in v:
            _validate_registered("capture_kinds", kind)
        return v
    sae_layer: Optional[int] = None
    overwrite: bool = False
    seed: int = 0
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

import copy
import importlib
import ml_collections as mlc


def set_inf(c, inf):
    for k, v in c.items():
        if isinstance(v, mlc.ConfigDict):
            set_inf(v, inf)
        elif k == "inf":
            c[k] = inf


def enforce_config_constraints(config):
    def string_to_setting(s):
        path = s.split('.')
        setting = config
        for p in path:
            setting = setting[p]

        return setting

    mutually_exclusive_bools = [
        (
            "model.template.average_templates",
            "model.template.offload_templates"
        ),
        (
            "globals.use_lma",
            "globals.use_flash",
            "globals.use_deepspeed_evo_attention"
        ),
    ]

    for options in mutually_exclusive_bools:
        option_settings = [string_to_setting(o) for o in options]
        if sum(option_settings) > 1:
            raise ValueError(f"Only one of {', '.join(options)} may be set at a time")

    fa_is_installed = importlib.util.find_spec("flash_attn") is not None
    if config.globals.use_flash and not fa_is_installed:
        raise ValueError("use_flash requires that FlashAttention is installed")

    deepspeed_is_installed = importlib.util.find_spec("deepspeed") is not None
    ds4s_is_installed = deepspeed_is_installed and importlib.util.find_spec(
        "deepspeed.ops.deepspeed4science") is not None
    if config.globals.use_deepspeed_evo_attention and not ds4s_is_installed:
        raise ValueError(
            "use_deepspeed_evo_attention requires that DeepSpeed be installed "
            "and that the deepspeed.ops.deepspeed4science package exists"
        )

    if (
            config.globals.offload_inference and
            not config.model.template.average_templates
    ):
        config.model.template.offload_templates = True


def model_config(
        name,
        train=False,
        low_prec=False,
        long_sequence_inference=False
):
    c = copy.deepcopy(config)
    # TRAINING PRESETS
    if name == "initial_training":
        # AF2 Suppl. Table 4, "initial training" setting
        pass
    elif name == "finetuning":
        # AF2 Suppl. Table 4, "finetuning" setting
        c.data.train.crop_size = 384
        c.data.train.max_extra_msa = 5120
        c.data.train.max_msa_clusters = 512
        c.loss.violation.weight = 1.
        c.loss.experimentally_resolved.weight = 0.01
    elif name == "finetuning_ptm":
        c.data.train.max_extra_msa = 5120
        c.data.train.crop_size = 384
        c.data.train.max_msa_clusters = 512
        c.loss.violation.weight = 1.
        c.loss.experimentally_resolved.weight = 0.01
        c.model.heads.tm.enabled = True
        c.loss.tm.weight = 0.1
    elif name == "finetuning_no_templ":
        # AF2 Suppl. Table 4, "finetuning" setting
        c.data.train.crop_size = 384
        c.data.train.max_extra_msa = 5120
        c.data.train.max_msa_clusters = 512
        c.model.template.enabled = False
        c.loss.violation.weight = 1.
        c.loss.experimentally_resolved.weight = 0.01
    elif name == "finetuning_no_templ_ptm":
        # AF2 Suppl. Table 4, "finetuning" setting
        c.data.train.crop_size = 384
        c.data.train.max_extra_msa = 5120
        c.data.train.max_msa_clusters = 512
        c.model.template.enabled = False
        c.loss.violation.weight = 1.
        c.loss.experimentally_resolved.weight = 0.01
        c.model.heads.tm.enabled = True
        c.loss.tm.weight = 0.1
    # INFERENCE PRESETS
    elif name == "model_1":
        # AF2 Suppl. Table 5, Model 1.1.1
        c.data.train.max_extra_msa = 5120
        c.data.predict.max_extra_msa = 5120
        c.data.common.reduce_max_clusters_by_max_templates = True
        c.data.common.use_templates = True
        c.data.common.use_template_torsion_angles = True
        c.model.template.enabled = True
    elif name == "model_2":
        # AF2 Suppl. Table 5, Model 1.1.2
        c.data.common.reduce_max_clusters_by_max_templates = True
        c.data.common.use_templates = True
        c.data.common.use_template_torsion_angles = True
        c.model.template.enabled = True
    elif name == "model_3":
        # AF2 Suppl. Table 5, Model 1.2.1
        c.data.train.max_extra_msa = 5120
        c.data.predict.max_extra_msa = 5120
        c.model.template.enabled = False
    elif name == "model_4":
        # AF2 Suppl. Table 5, Model 1.2.2
        c.data.train.max_extra_msa = 5120
        c.data.predict.max_extra_msa = 5120
        c.model.template.enabled = False
    elif name == "model_5":
        # AF2 Suppl. Table 5, Model 1.2.3
        c.model.template.enabled = False
    elif name == "model_1_ptm":
        c.data.train.max_extra_msa = 5120
        c.data.predict.max_extra_msa = 5120
        c.data.common.reduce_max_clusters_by_max_templates = True
        c.data.common.use_templates = True
        c.data.common.use_template_torsion_angles = True
        c.model.template.enabled = True
        c.model.heads.tm.enabled = True
        c.loss.tm.weight = 0.1
    elif name == "model_2_ptm":
        c.data.common.reduce_max_clusters_by_max_templates = True
        c.data.common.use_templates = True
        c.data.common.use_template_torsion_angles = True
        c.model.template.enabled = True
        c.model.heads.tm.enabled = True
        c.loss.tm.weight = 0.1
    elif name == "model_3_ptm":
        c.data.train.max_extra_msa = 5120
        c.data.predict.max_extra_msa = 5120
        c.model.template.enabled = False
        c.model.heads.tm.enabled = True
        c.loss.tm.weight = 0.1
    elif name == "model_4_ptm":
        c.data.train.max_extra_msa = 5120
        c.data.predict.max_extra_msa = 5120
        c.model.template.enabled = False
        c.model.heads.tm.enabled = True
        c.loss.tm.weight = 0.1
    elif name == "model_5_ptm":
        c.model.template.enabled = False
        c.model.heads.tm.enabled = True
        c.loss.tm.weight = 0.1
    # SINGLE SEQUENCE EMBEDDING PRESETS
    elif name == "seqemb_initial_training":
        c.data.train.max_msa_clusters = 1
        c.data.eval.max_msa_clusters = 1
        c.data.train.block_delete_msa = False
        c.data.train.max_distillation_msa_clusters = 1
    elif name == "seqemb_finetuning":
        c.data.train.max_msa_clusters = 1
        c.data.eval.max_msa_clusters = 1
        c.data.train.block_delete_msa = False
        c.data.train.max_distillation_msa_clusters = 1
        c.data.train.crop_size = 384
        c.loss.violation.weight = 1.
        c.loss.experimentally_resolved.weight = 0.01
    elif name == "seq_model_esm1b":
        c.data.common.use_templates = True
        c.data.common.use_template_torsion_angles = True
        c.model.template.enabled = True
        c.data.predict.max_msa_clusters = 1
    elif name == "seq_model_esm1b_ptm":
        c.data.common.use_templates = True
        c.data.common.use_template_torsion_angles = True
        c.model.template.enabled = True
        c.data.predict.max_msa_clusters = 1
        c.model.heads.tm.enabled = True
        c.loss.tm.weight = 0.1
    else:
        raise ValueError("Invalid model name")

    if name.startswith("seq"):
        # Tell the data pipeline that we will use sequence embeddings instead of MSAs.
        c.data.seqemb_mode.enabled = True
        c.globals.seqemb_mode_enabled = True
        # In seqemb mode, we turn off the ExtraMSAStack and Evoformer's column attention.
        c.model.extra_msa.enabled = False
        c.model.evoformer_stack.no_column_attention = True
        c.update(seq_mode_config.copy_and_resolve_references())

    if long_sequence_inference:
        assert (not train)
        c.globals.offload_inference = True
        # Default to DeepSpeed memory-efficient attention kernel unless use_lma is explicitly set
        c.globals.use_deepspeed_evo_attention = True if not c.globals.use_lma else False
        c.globals.use_flash = False
        c.model.template.offload_inference = True
        c.model.template.template_pair_stack.tune_chunk_size = False
        c.model.extra_msa.extra_msa_stack.tune_chunk_size = False
        c.model.evoformer_stack.tune_chunk_size = False

    if train:
        c.globals.blocks_per_ckpt = 1
        c.globals.chunk_size = None
        c.globals.use_lma = False
        c.globals.offload_inference = False
        c.model.template.average_templates = False
        c.model.template.offload_templates = False

    if low_prec:
        c.globals.eps = 1e-4
        # If we want exact numerical parity with the original, inf can't be
        # a global constant
        set_inf(c, 1e4)

    enforce_config_constraints(c)

    return c


c_z = mlc.FieldReference(128, field_type=int)
c_m = mlc.FieldReference(256, field_type=int)
c_t = mlc.FieldReference(64, field_type=int)
c_e = mlc.FieldReference(64, field_type=int)
c_s = mlc.FieldReference(1024, field_type=int)  # 384

# For seqemb mode, dimension size of the per-residue sequence embedding passed to the model
# In current model, the dimension size is the ESM-1b dimension size i.e. 1280.
preemb_dim_size = mlc.FieldReference(1280, field_type=int)

blocks_per_ckpt = mlc.FieldReference(None, field_type=int)
chunk_size = mlc.FieldReference(4, field_type=int)
aux_distogram_bins = mlc.FieldReference(64, field_type=int)
tm_enabled = mlc.FieldReference(False, field_type=bool)
eps = mlc.FieldReference(1e-8, field_type=float)
templates_enabled = mlc.FieldReference(False, field_type=bool)
embed_template_torsion_angles = mlc.FieldReference(True, field_type=bool)
tune_chunk_size = mlc.FieldReference(True, field_type=bool)

NUM_RES = "num residues placeholder"
NUM_MSA_SEQ = "msa placeholder"
NUM_EXTRA_SEQ = "extra msa placeholder"
NUM_TEMPLATES = "num templates placeholder"

config = mlc.ConfigDict(
    {
        "data": {
            "common": {
                "feat": {
                    "aatype": [NUM_RES],
                    "all_atom_mask": [NUM_RES, None],
                    "all_atom_mask_true": [NUM_RES, None],
                    "all_atom_positions": [NUM_RES, None, None],
                    "alt_chi_angles": [NUM_RES, None],
                    "atom14_alt_gt_exists": [NUM_RES, None],
                    "atom14_alt_gt_positions": [NUM_RES, None, None],
                    "atom14_atom_exists": [NUM_RES, None],
                    "atom14_atom_is_ambiguous": [NUM_RES, None],
                    "atom14_gt_exists": [NUM_RES, None],
                    "atom14_gt_positions": [NUM_RES, None, None],
                    "atom37_atom_exists": [NUM_RES, None],
                    "backbone_rigid_mask": [NUM_RES],
                    "backbone_rigid_tensor": [NUM_RES, None, None],
                    "bert_mask": [NUM_MSA_SEQ, NUM_RES],
                    "chi_angles_sin_cos": [NUM_RES, None, None],
                    "chi_mask": [NUM_RES, None],
                    "extra_deletion_value": [NUM_EXTRA_SEQ, NUM_RES],
                    "extra_has_deletion": [NUM_EXTRA_SEQ, NUM_RES],
                    "extra_msa": [NUM_EXTRA_SEQ, NUM_RES],
                    "extra_msa_mask": [NUM_EXTRA_SEQ, NUM_RES],
                    "extra_msa_row_mask": [NUM_EXTRA_SEQ],
                    "is_distillation": [],
                    "msa_feat": [NUM_MSA_SEQ, NUM_RES, None],
                    "msa_mask": [NUM_MSA_SEQ, NUM_RES],
                    "msa_row_mask": [NUM_MSA_SEQ],
                    "no_recycling_iters": [],
                    "pseudo_beta": [NUM_RES, None],
                    "pseudo_beta_mask": [NUM_RES],
                    "residue_index": [NUM_RES],
                    "residx_atom14_to_atom37": [NUM_RES, None],
                    "residx_atom37_to_atom14": [NUM_RES, None],
                    "resolution": [],
                    "rigidgroups_alt_gt_frames": [NUM_RES, None, None, None],
                    "rigidgroups_group_exists": [NUM_RES, None],
                    "rigidgroups_group_is_ambiguous": [NUM_RES, None],
                    "rigidgroups_gt_exists": [NUM_RES, None],
                    "rigidgroups_gt_frames": [NUM_RES, None, None, None],
                    "seq_length": [],
                    "seq_mask": [NUM_RES],
                    "target_feat": [NUM_RES, None],
                    "template_aatype": [NUM_TEMPLATES, NUM_RES],
                    "template_all_atom_mask": [NUM_TEMPLATES, NUM_RES, None],
                    "template_all_atom_positions": [
                        NUM_TEMPLATES, NUM_RES, None, None,
                    ],
                    "template_alt_torsion_angles_sin_cos": [
                        NUM_TEMPLATES, NUM_RES, None, None,
                    ],
                    "template_backbone_rigid_mask": [NUM_TEMPLATES, NUM_RES],
                    "template_backbone_rigid_tensor": [
                        NUM_TEMPLATES, NUM_RES, None, None,
                    ],
                    "template_mask": [NUM_TEMPLATES],
                    "template_pseudo_beta": [NUM_TEMPLATES, NUM_RES, None],
                    "template_pseudo_beta_mask": [NUM_TEMPLATES, NUM_RES],
                    "template_sum_probs": [NUM_TEMPLATES, None],
                    "template_torsion_angles_mask": [
                        NUM_TEMPLATES, NUM_RES, None,
                    ],
                    "template_torsion_angles_sin_cos": [
                        NUM_TEMPLATES, NUM_RES, None, None,
                    ],
                    "true_msa": [NUM_MSA_SEQ, NUM_RES],
                    "use_clamped_fape": [],
                },
                "block_delete_msa": {
                    "msa_fraction_per_block": 0.3,
                    "randomize_num_blocks": False,
                    "num_blocks": 5,
                },
                "masked_msa": {
                    "profile_prob": 0.1,
                    "same_prob": 0.1,
                    "uniform_prob": 0.1,
                },
                "max_recycling_iters": 0,
                "msa_cluster_features": True,
                "reduce_msa_clusters_by_max_templates": False,
                "resample_msa_in_recycling": True,
                "template_features": [
                    "template_all_atom_positions",
                    "template_sum_probs",
                    "template_aatype",
                    "template_all_atom_mask",
                ],
                "unsupervised_features": [
                    "aatype",
                    "residue_index",
                    "msa",
                    "num_alignments",
                    "seq_length",
                    "between_segment_residues",
                    "deletion_matrix",
                    "no_recycling_iters",
                ],
                "use_templates": templates_enabled,
                "use_template_torsion_angles": embed_template_torsion_angles,
            },
            "seqemb_mode": {  # Configuration for sequence embedding mode
                "enabled": False,  # If True, use seq emb instead of MSA
            },
            "supervised": {
                "clamp_prob": 0.9,
                "supervised_features": [
                    "all_atom_mask",
                    "all_atom_mask_true",
                    "all_atom_positions",
                    "resolution",
                    "use_clamped_fape",
                    "is_distillation",
                ],
            },
            "predict": {
                "fixed_size": True,
                "subsample_templates": False,  # We want top templates.
                "block_delete_msa": False,
                "masked_msa_replace_fraction": 0.15,
                "max_msa_clusters": 512,
                "max_extra_msa": 1024,
                "max_template_hits": 4,
                "max_templates": 4,
                "crop": False,
                "crop_size": None,
                "supervised": False,
                "uniform_recycling": False,
            },
            "eval": {
                "fixed_size": True,
                "subsample_templates": False,  # We want top templates.
                "block_delete_msa": False,
                "masked_msa_replace_fraction": 0.15,
                "max_msa_clusters": 128,
                "max_extra_msa": 1024,
                "max_template_hits": 4,
                "max_templates": 4,
                "crop": False,
                "crop_size": None,
                "supervised": True,
                "uniform_recycling": False,
            },
            "train": {
                "fixed_size": True,
                "subsample_templates": True,
                "block_delete_msa": True,
                "masked_msa_replace_fraction": 0.15,
                "max_msa_clusters": 128,
                "max_extra_msa": 1024,
                "max_template_hits": 4,
                "max_templates": 4,
                "shuffle_top_k_prefiltered": 20,
                "crop": True,
                "crop_size": 256,
                "supervised": True,
                "clamp_prob": 0.9,
                "max_distillation_msa_clusters": 1000,
                "uniform_recycling": True,
                "distillation_prob": 0.75,
            },
            "data_module": {
                "use_small_bfd": False,
                "data_loaders": {
                    "batch_size": 1,
                    "num_workers": 16,
                    "pin_memory": True,
                },
            },
        },
        # Recurring FieldReferences that can be changed globally here
        "globals": {
            "seqemb_mode_enabled": False,  # Global flag for enabling seq emb mode
            "blocks_per_ckpt": blocks_per_ckpt,
            "chunk_size": chunk_size,
            # Use DeepSpeed memory-efficient attention kernel. Mutually
            # exclusive with use_lma and use_flash.
            "use_deepspeed_evo_attention": False,
            # Use Staats & Rabe's low-memory attention algorithm. Mutually
            # exclusive with use_deepspeed_evo_attention and use_flash.
            "use_lma": False,
            # Use FlashAttention in selected modules. Mutually exclusive with
            # use_deepspeed_evo_attention and use_lma. Doesn't work that well
            # on long sequences (>1000 residues).
            "use_flash": False,
            "offload_inference": False,
            "c_z": c_z,
            "c_m": c_m,
            "c_t": c_t,
            "c_e": c_e,
            "c_s": c_s,
            "eps": eps,
        },
        "model": {
            "_mask_trans": False,
            "input_embedder": {
                "tf_dim": 22,
                "msa_dim": 49,
                "c_z": c_z,
                "c_m": c_m,
                "relpos_k": 32,
            },
            "recycling_embedder": {
                "c_z": c_z,
                "c_m": c_m,
                "min_bin": 3.25,
                "max_bin": 20.75,
                "no_bins": 15,
                "inf": 1e8,
            },
            "template": {
                "distogram": {
                    "min_bin": 3.25,
                    "max_bin": 50.75,
                    "no_bins": 39,
                },
                "template_angle_embedder": {
                    # DISCREPANCY: c_in is supposed to be 51.
                    "c_in": 57,
                    "c_out": c_m,
                },
                "template_pair_embedder": {
                    "c_in": 88,
                    "c_out": c_t,
                },
                "template_pair_stack": {
                    "c_t": c_t,
                    # DISCREPANCY: c_hidden_tri_att here is given in the supplement
                    # as 64. In the code, it's 16.
                    "c_hidden_tri_att": 16,
                    "c_hidden_tri_mul": 64,
                    "no_blocks": 2,
                    "no_heads": 4,
                    "pair_transition_n": 2,
                    "dropout_rate": 0.25,
                    "blocks_per_ckpt": blocks_per_ckpt,
                    "tune_chunk_size": tune_chunk_size,
                    "inf": 1e9,
                },
                "template_pointwise_attention": {
                    "c_t": c_t,
                    "c_z": c_z,
                    # DISCREPANCY: c_hidden here is given in the supplement as 64.
                    # It's actually 16.
                    "c_hidden": 16,
                    "no_heads": 4,
                    "inf": 1e5,  # 1e9,
                },
                "inf": 1e5,  # 1e9,
                "eps": eps,  # 1e-6,
                "enabled": templates_enabled,
                "embed_angles": embed_template_torsion_angles,
                "use_unit_vector": False,
                # Approximate template computation, saving memory.
                # In our experiments, results are equivalent to or better than
                # the stock implementation. Should be enabled for all new
                # training runs.
                "average_templates": False,
                # Offload template embeddings to CPU memory. Vastly reduced
                # memory consumption at the cost of a modest increase in
                # runtime. Useful for inference on very long sequences.
                # Mutually exclusive with average_templates. Automatically
                # enabled if offload_inference is set.
                "offload_templates": False,
            },
            "extra_msa": {
                "extra_msa_embedder": {
                    "c_in": 25,
                    "c_out": c_e,
                },
                "extra_msa_stack": {
                    "c_m": c_e,
                    "c_z": c_z,
                    "c_hidden_msa_att": 8,
                    "c_hidden_opm": 32,
                    "c_hidden_mul": 128,
                    "c_hidden_pair_att": 32,
                    "no_heads_msa": 8,
                    "no_heads_pair": 4,
                    "no_blocks": 4,
                    "transition_n": 4,
                    "msa_dropout": 0.15,
                    "pair_dropout": 0.25,
                    "clear_cache_between_blocks": False,
                    "tune_chunk_size": tune_chunk_size,
                    "inf": 1e9,
                    "eps": eps,  # 1e-10,
                    "ckpt": blocks_per_ckpt is not None,
                },
                "enabled": True,
            },
            "evoformer_stack": {
                "c_m": c_m,
                "c_z": c_z,
                "c_hidden_msa_att": 32,
                "c_hidden_opm": 32,
                "c_hidden_mul": 128,
                "c_hidden_pair_att": 32,
                "c_s": c_s,
                "no_heads_msa": 8,
                "no_heads_pair": 4,
                "no_blocks": 48,
                "transition_n": 4,
                "msa_dropout": 0.15,
                "pair_dropout": 0.25,
                "no_column_attention": False,
                "blocks_per_ckpt": blocks_per_ckpt,
                "clear_cache_between_blocks": False,
                "tune_chunk_size": tune_chunk_size,
                "inf": 1e9,
                "eps": eps,  # 1e-10,
            },
            "structure_module": {
                "c_s": c_s,
                "c_z": c_z,
                "c_ipa": 16,
                "c_resnet": 128,
                "no_heads_ipa": 12,
                "no_qk_points": 4,
                "no_v_points": 8,
                "dropout_rate": 0.1,
                "no_blocks": 8,
                "no_transition_layers": 1,
                "no_resnet_blocks": 2,
                "no_angles": 7,
                "trans_scale_factor": 10,
                "epsilon": eps,  # 1e-12,
                "inf": 1e5,
            },
            "heads": {
                "lddt": {
                    "no_bins": 50,
                    "c_in": c_s,
                    "c_hidden": 128,
                },
                "distogram": {
                    "c_z": c_z,
                    "no_bins": aux_distogram_bins,
                },
                "tm": {
                    "c_z": c_z,
                    "no_bins": aux_distogram_bins,
                    "enabled": tm_enabled,
                },
                "masked_msa": {
                    "c_m": c_m,
                    "c_out": 23,
                },
                "experimentally_resolved": {
                    "c_s": c_s,
                    "c_out": 37,
                },
            },
        },
        "relax": {
            "max_iterations": 0,  # no max
            "tolerance": 2.39,
            "stiffness": 10.0,
            "max_outer_iterations": 20,
            "exclude_residues": [],
        },
        "loss": {
            "distogram": {
                "min_bin": 2.3125,
                "max_bin": 21.6875,
                "no_bins": 64,
                "eps": eps,  # 1e-6,
                "weight": 0.3,
            },
            "experimentally_resolved": {
                "eps": eps,  # 1e-8,
                "min_resolution": 0.1,
                "max_resolution": 3.0,
                "weight": 0.0,
            },
            "fape": {
                "backbone": {
                    "clamp_distance": 10.0,
                    "loss_unit_distance": 10.0,
                    "weight": 0.5,
                },
                "sidechain": {
                    "clamp_distance": 10.0,
                    "length_scale": 10.0,
                    "weight": 0.5,
                },
                "eps": 1e-4,
                "weight": 1.0,
            },
            "plddt_loss": {
                "min_resolution": 0.1,
                "max_resolution": 3.0,
                "cutoff": 15.0,
                "no_bins": 50,
                "eps": eps,  # 1e-10,
                "weight": 0.01,
            },
            "masked_msa": {
                "eps": eps,  # 1e-8,
                "weight": 2.0,
            },
            "supervised_chi": {
                "chi_weight": 0.5,
                "angle_norm_weight": 0.01,
                "eps": eps,  # 1e-6,
                "weight": 1.0,
            },
            "violation": {
                "violation_tolerance_factor": 12.0,
                "clash_overlap_tolerance": 1.5,
                "eps": eps,  # 1e-6,
                "weight": 0.0,
            },
            "tm": {
                "max_bin": 31,
                "no_bins": 64,
                "min_resolution": 0.1,
                "max_resolution": 3.0,
                "eps": eps,  # 1e-8,
                "weight": 0.,
                "enabled": tm_enabled,
            },
            "eps": eps,
        },
        "ema": {"decay": 0.999},
    }
)

seq_mode_config = mlc.ConfigDict({
    "data": {
        "common": {
            "feat": {
                "seq_embedding": [NUM_RES, None],
            },
            "seqemb_features": [  # List of features to be generated in seqemb mode
                "seq_embedding"
            ],
        },
        "seqemb_mode": {  # Configuration for sequence embedding mode
            "enabled": True,  # If True, use seq emb instead of MSA
        },
    },
    "globals": {
        "seqemb_mode_enabled": True,
    },
    "model": {
        "preembedding_embedder": {  # Used in sequence embedding mode
            "tf_dim": 22,
            "preembedding_dim": preemb_dim_size,
            "c_z": c_z,
            "c_m": c_m,
            "relpos_k": 32,
        },
    }
})

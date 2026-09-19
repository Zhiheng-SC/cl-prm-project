# Derived metrics
OVM_MINUS_SELF_CONSISTENCY = "ovm_minus_self_consistency_accuracy"
OVM_MINUS_GREEDY = "ovm_minus_greedy_accuracy"
OVM_MINUS_ORM = "ovm_minus_orm_accuracy"
EXPANSION_MINUS_VANILLA_CORRECT = "last_expansion_minus_vanilla_correct_fraction"
OVM_SAMPLING_RUNTIME_RATIO = "ovm_to_sampling_runtime_ratio"

# Accuracy metrics
GREEDY_ACC = "greedy_accuracy"
SELF_CONSISTENCY_ACC = "self_consistency_accuracy"
ORM_ACC = "orm_accuracy"
OVM_ACC = "ovm_accuracy"

# Quality metrics
VANILLA_CORRECT_FRACTION = "vanilla_correct_fraction"
OVM_LAST_EXPANSION_CORRECT_FRACTION = "ovm_last_expansion_correct_fraction"
OVM_FINAL_BEAM_CORRECT_FRACTION = "ovm_final_beam_correct_fraction"

# Memory metrics
MAX_PEAK_ALLOCATED_GIB = "maximum_peak_allocated_gib"
MAX_PEAK_RESERVED_GIB = "maximum_peak_reserved_gib"

# Runtime metrics
GREEDY_RUNTIME_SEC = "greedy_runtime_seconds"
SAMPLING_RUNTIME_SEC = "sampling_runtime_seconds"
ORM_RUNTIME_SEC = "orm_runtime_seconds"
OVM_RUNTIME_SEC = "ovm_runtime_seconds"
"""Fixed BM128 eager numerical plan/gates; all imports are stdlib CPU."""
import biv150_konly_2d128_compile_contract as compile_contract
import biv150_konly_q32_numeric_contract as old


def plan():
    p=compile_contract.plan()
    # Empty-allocation compile observations must not become eager-run claims.
    for name in ('kernel_launches','initialized_tensor_elements','metadata_values_initialized'):
        p.pop(name)
    p.update(compile_only=False,metadata_initialization_required=True,
        attention_launches_planned=4,encoder_launches_planned=0,patterns=[0,1],
        CPU_encoded_cache=True,Graph_verified=False,side_stream_verified=False,
        actual_I8_operand_hoist_verified=False,timing_enabled=False)
    return p


def gates(baseline,candidate,original,quantized):
    g=old.gates(baseline,candidate,original,quantized)
    q=g['quantized_operand'];b=g['original_BF16']
    g.update(quantized_implementation_passed=g['baseline_hard_passed'] and q['finite'] and q['pointwise_passed'],
             original_BF16_quality_passed=b['finite'] and b['relative_rmse']<=g['original_relative_rmse_limit'])
    return g


def policy(hard,numeric,diagnostic):
    names={'finite','baseline_reference','CPU_codec_cache_bytes','output_canaries',
           'read_only_inputs','actual_IR','quantized_implementation'}
    if set(hard)!=names or any(v is not True for v in hard.values()):
        raise ValueError('Eager hard failure cannot be bypassed')
    if type(numeric) is not bool or type(diagnostic) is not bool:raise ValueError('Explicit gate booleans')
    if not numeric and not diagnostic:raise ValueError('Original BF16 quality gate failed; explicit diagnostic required')
    return dict(component_requested_hard_gates_passed=True,original_BF16_quality_passed=numeric,
        numeric_diagnostic=diagnostic,precision_gate=False,full_numerical_matrix_verified=False,
        model_or_service_verified=False,Graph_verified=False,side_stream_verified=False,performance_measured=False)

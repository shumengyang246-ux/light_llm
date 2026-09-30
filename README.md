# light_llm
一个基于Triton算子和paged attention的轻量级大模型推理框架。

A small decoder-only inference framework using the bundled Triton kernels.
The verified configuration is Qwen2.5-Math-1.5B, BF16, one A100, and the existing
`/opt/conda/envs/llm` environment. Start with eager execution and a bounded KV cache.

## Quick start

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate llm
cd /root/ysm/Triton/light_llm-main
CUDA_VISIBLE_DEVICES=0 python examples/run_generation.py
```

Equivalent without shell activation:

```bash
CUDA_VISIBLE_DEVICES=0 /opt/conda/envs/llm/bin/python examples/run_generation.py
```

```python
from light_llm.engine.llm import LLM, SamplingParams

llm = LLM(
    '/root/ysm/models/Qwen2.5-Math-1.5B',
    device='cuda', use_cuda_graph=False,
    max_seq_len=512, max_gpu_num_blocks=1024,
)
params = SamplingParams(max_gen_len=32, temperature=0.0)
for result in llm.generate('What is 2 + 3?', params):
    print(result.text)
```

`SamplingParams` uses `max_gen_len`, not `max_tokens`. `max_gpu_num_blocks`
currently means the number of token slots in this implementation, not pages.
With no explicit limit the runner profiles and allocates a much larger KV pool.
The model is a base math model, not an instruction-tuned chat model.

## Terminal chat

Run `python examples/run_generation.py` to chat interactively. The model loads
once and streams each reply. Conversation history uses the model's chat template.
Use `/clear` to reset history, `/exit` or `/quit` to leave, and `/help` for commands.
Ctrl+C cancels a running reply; at the input prompt it exits.

```bash
CUDA_VISIBLE_DEVICES=0 python examples/run_generation.py --output outputs/chat.json
python examples/run_generation.py --help
```

See [examples/README.md](examples/README.md) for system prompts, sampling and
memory settings. Old complete turns are dropped when the context is full.
The terminal chat default is the existing Qwen2.5-7B-Instruct checkpoint, with
temperature 0.6. The former Qwen2.5-Math-1.5B base checkpoint repeats under chat
prompts in both this engine and Transformers; having a chat template does not
make a base checkpoint instruction-tuned.
The former `--prompt` and `--stream` options are replaced by interactive chat.

## Validation

```bash
CUDA_VISIBLE_DEVICES=0 python -m pytest -q tests/test_dense_kernels.py
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -m scripts.validate_qwen2 --cuda-graph --report docs/validation_20260928.json
```

The kernel regressions cover FP16/BF16 residual RMSNorm, unequal-length prefill,
GQA decode with permuted cache slots/request ids, and KV writes. The integration
script loads all 28 layers, compares logits and greedy generation with
Transformers SDPA, checks repeated calls and streaming, and checks eager/graph
parity for batch sizes 1 and 2 with sequence buckets 64 and 128.

For teacher-forced logits the declared bounds are relative RMSE < 2% and
per-token cosine similarity > 0.999; single-request top-1 agreement must be at
least 95%, and the selected single-request greedy examples must match exactly.
Batch logits are checked using the same bounds. Exact batch token equality with
Transformers is reported separately: different BF16 fusion/GEMM shapes can alter
near-tied token choices and EOS, so it is not asserted. See the saved JSON for
actual results, including those differences. This is a smoke/regression check,
not a model-quality or performance benchmark.

Transformers eager Attention rounds intermediate attention scores in BF16;
SDPA is used as the reference for these fused attention kernels. This distinction
was measured on the actual checkpoint, not inferred from successful text output.

## Current scope

- Verified: single-GPU Qwen2 dense inference, greedy generation, eager and the
  small CUDA Graph configuration described above.
- Existing Llama and advanced parallel/quantization/multimodal code is not an
  end-to-end validated feature set. Several advanced implementations are absent.
- The bundled Attention, RoPE, residual RMSNorm, SwiGLU and KV update operations
  use Triton. Unquantized Linear still uses PyTorch/cuBLAS; single-GPU embedding
  and sampling also use PyTorch. This is not yet an all-Triton implementation.
- Optional FlashInfer/FlashMLA/DeepGEMM adapters are not required or registered in
  the dense startup path. The project does not import the `rapid_llm` package.

## Repair provenance (2026-09-28)

Missing `modules/quantization/kv_cache.py`, `modules/quantization/utils.py`,
`kernels/ops/quantization/scale_layout.py`, `kernels/ops/quantization/w8a16.py`,
and `kernels/ops/tile_policy.py` were copied from the user's local `rapid_llm-main`.
The complete `kernels/ops/layernorm/skip_rmsnorm.py` was taken from the same
reference, then residual-add rounding was aligned with the checkpoint's dtype.
Imports were adapted to `light_llm`. The dense model no longer requires a missing
MoE class just to import/load; kernel registrations are limited to shipped dense
implementations. Missing decode constants and the smoke script API argument were
also corrected.

The original server project is backed up at:
`/root/ysm/Triton/light_llm-main.backup-20260928-163446.tar.gz`.
No package installation or environment upgrade was performed during this repair.


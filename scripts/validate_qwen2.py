"""Validate the real Qwen2 checkpoint against Transformers on one CUDA device.

Run from the project root: python -m scripts.validate_qwen2 --report docs/validation.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from light_llm.engine.llm import LLM, SamplingParams
from light_llm.kernels.dispatcher import REGISTRY


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='/root/ysm/models/Qwen2.5-Math-1.5B')
    parser.add_argument('--report', type=Path)
    parser.add_argument('--cuda-graph', action='store_true')
    args = parser.parse_args()
    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats()
    engine = LLM(args.model, device='cuda', use_cuda_graph=False,
                 max_seq_len=512, max_gpu_num_blocks=2048)
    params = SamplingParams(max_gen_len=16, temperature=0.0,
                            repetition_penalty=1.0, stop_on_repeat=False, logprobs=0)
    prompts = ['What is 2 + 3?', 'The capital of France is',
               'Solve the equation x + 7 = 12. The value of x is']
    records = []
    for prompt in prompts:
        logits = []
        def capture(module, inputs, output):
            logits.append(output.reshape(-1, output.shape[-1])[-1].detach().float().cpu())
        handle = engine.model_runner.model.register_forward_hook(capture)
        try:
            output = engine.generate(prompt, params)[0]
        finally:
            handle.remove()
        ids = [entry.token_id for entry in output.outputs[0].logprobs]
        assert ids and output.text, 'Empty generation'
        records.append(dict(prompt=prompt, text=output.text, ids=ids, logits=logits[:len(ids)]))
        print('GENERATED', repr(prompt), '->', repr(output.text), flush=True)

    batch_step_logits = []
    def capture_batch(module, inputs, output):
        batch_step_logits.append(output.reshape(2, -1, output.shape[-1])[:, -1].detach().float().cpu())
    handle = engine.model_runner.model.register_forward_hook(capture_batch)
    try:
        batch = engine.generate(prompts[:2], params)
    finally:
        handle.remove()
    for i, output in enumerate(batch):
        assert output.text and output.outputs[0].logprobs
        print('BATCH_GENERATED', repr(output.prompt), '->', repr(output.text), flush=True)
    repeated = engine.generate(prompts[0], params)[0]
    assert repeated.text == records[0]['text'], 'Repeated call differs'
    free_after_repeat = engine.model_runner.kv_cache_manager.can_use_mem_size
    repeated = engine.generate(prompts[0], params)[0]
    assert engine.model_runner.kv_cache_manager.can_use_mem_size == free_after_repeat, 'KV capacity leaked'
    streamed = ''.join(chunk[0] for chunk in engine.stream(prompts[0], params))
    assert streamed == records[0]['text'], 'Streaming differs from non-streaming'
    peak_light = torch.cuda.max_memory_allocated() / 1024**3
    print('BATCH_REPEAT_STREAM_OK', flush=True)

    hf = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
        local_files_only=True, attn_implementation='sdpa').to('cuda').eval()
    comparisons = []
    matched, count = 0, 0
    with torch.inference_mode():
        for row in records:
            prompt_ids = engine.tokenizer.encode(row['prompt'], add_special_tokens=True)
            # Teacher forcing compares every decode step on exactly the same history.
            all_ids = torch.tensor([prompt_ids + row['ids'][:-1]], device='cuda')
            ref = hf(all_ids, use_cache=False).logits[0, len(prompt_ids)-1:].float().cpu()
            actual = torch.stack(row['logits'])
            assert actual.shape == ref.shape
            assert torch.isfinite(actual).all()
            relative_rmse = ((actual-ref).square().mean().sqrt() / ref.square().mean().sqrt()).item()
            cosine = F.cosine_similarity(actual, ref, dim=-1).min().item()
            top1_matches = int((actual.argmax(-1) == ref.argmax(-1)).sum())
            matched += top1_matches; count += ref.shape[0]
            assert relative_rmse < 0.02, ('Logit relative RMSE', relative_rmse)
            assert cosine > 0.999, ('Logit cosine', cosine)

            prompt_tensor = torch.tensor([prompt_ids], device='cuda')
            hf_ids = hf.generate(prompt_tensor, attention_mask=torch.ones_like(prompt_tensor),
                max_new_tokens=16, do_sample=False, repetition_penalty=1.0,
                pad_token_id=engine.pad_id, eos_token_id=sorted(engine.stop_token_ids))[0, len(prompt_ids):].tolist()
            # Engine omits EOS in its output records/text; normalise both sequences.
            hf_visible = []
            for token in hf_ids:
                if token in engine.stop_token_ids: break
                hf_visible.append(token)
            comparison = dict(prompt=row['prompt'], light_text=row['text'],
                hf_text=engine.tokenizer.decode(hf_visible, skip_special_tokens=True),
                exact_greedy_tokens=row['ids'] == hf_visible,
                teacher_forced_top1_matches=top1_matches, tokens=ref.shape[0],
                relative_rmse=relative_rmse, min_cosine=cosine,
                max_abs_logit_error=(actual-ref).abs().max().item())
            comparisons.append(comparison)
            print('REFERENCE', json.dumps(comparison, ensure_ascii=False), flush=True)
    assert matched/count >= 0.95, ('Teacher-forced top1 agreement', matched/count)
    assert all(r['exact_greedy_tokens'] for r in comparisons), 'Single greedy generation differs from HF SDPA'
    tokenizer = engine.tokenizer
    old_side = tokenizer.padding_side
    tokenizer.padding_side = 'left'
    try:
        inputs = tokenizer(prompts[:2], return_tensors='pt', padding=True).to('cuda')
    finally:
        tokenizer.padding_side = old_side
    with torch.inference_mode():
        hf_batch = hf.generate(**inputs, max_new_tokens=16, do_sample=False,
            repetition_penalty=1.0, pad_token_id=engine.pad_id, eos_token_id=sorted(engine.stop_token_ids))
    batch_comparisons = []
    for i, row in enumerate(hf_batch[:, inputs['input_ids'].shape[1]:].tolist()):
        visible = []
        for token in row:
            if token in engine.stop_token_ids: break
            visible.append(token)
        actual = [r.token_id for r in batch[i].outputs[0].logprobs]
        prompt_ids = tokenizer.encode(prompts[i], add_special_tokens=True)
        with torch.inference_mode():
            reference = hf(torch.tensor([prompt_ids + actual[:-1]], device='cuda'), use_cache=False).logits[0, len(prompt_ids)-1:].float().cpu()
        observed = torch.stack([step[i] for step in batch_step_logits[:len(actual)]])
        relative_rmse = ((observed-reference).square().mean().sqrt() / reference.square().mean().sqrt()).item()
        cosine = F.cosine_similarity(observed, reference, dim=-1).min().item()
        assert relative_rmse < 0.02, ('Batch logits RMSE', relative_rmse)
        assert cosine > 0.999, ('Batch logits cosine', cosine)
        result = dict(relative_rmse=relative_rmse, min_cosine=cosine, prompt=prompts[i], light_text=batch[i].text, hf_text=tokenizer.decode(visible, skip_special_tokens=True), exact_greedy_tokens=actual == visible)
        batch_comparisons.append(result)
        print('BATCH_REFERENCE', json.dumps(result, ensure_ascii=False), flush=True)
    del hf
    torch.cuda.empty_cache()
    graph_ok = None
    if args.cuda_graph:
        engine.model_runner.enable_cuda_graph(batch_sizes=(1, 2), seq_len_buckets=(64, 128))
        assert engine.model_runner._graph_manager is not None, 'Graph capture silently fell back'
        graphed = engine.generate(prompts[:2], params)
        for i, output in enumerate(graphed):
            assert [r.token_id for r in output.outputs[0].logprobs] == [r.token_id for r in batch[i].outputs[0].logprobs], 'CUDA graph differs from eager batch'
        graph_ok = True
        print('CUDA_GRAPH_OK', flush=True)
    report = dict(model=args.model, torch=torch.__version__, gpu=torch.cuda.get_device_name(),
        layers=engine.model_runner.config.num_layers, batch_repeat_stream=True,
        reference_attention='sdpa', batch_comparisons=batch_comparisons,
        cuda_graph=graph_ok, peak_light_allocated_gib=peak_light,
        teacher_forced_top1_agreement=matched/count, comparisons=comparisons,
        dispatch=[dict(op=d.key.op, target=d.spec.target) for d in REGISTRY.decisions()])
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print('VALIDATION_PASSED', flush=True)


if __name__ == '__main__':
    main()

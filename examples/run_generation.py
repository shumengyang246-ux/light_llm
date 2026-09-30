"""Interactive streaming terminal chat: python examples/run_generation.py."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_MODEL = "/root/ysm/models/Qwen2.5-7B-Instruct"
HELP = "/clear 清空历史 | /exit 或 /quit 退出 | /help 显示帮助\n每次输入一行，回车发送；回复时 Ctrl+C 取消本轮，输入时 Ctrl+C 退出。"


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="light_llm 终端多轮对话，默认流式显示回复。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=Path, default=Path(DEFAULT_MODEL), help="本地模型目录")
    parser.add_argument("--system", help="自定义系统提示，省略时使用模型聊天模板的默认设置")
    parser.add_argument("--max-new-tokens", type=positive_int, default=256, help="每轮最多生成的 token 数")
    parser.add_argument("--max-seq-len", type=positive_int, default=2048, help="历史、输入与输出的上下文上限")
    parser.add_argument("--kv-cache-tokens", type=positive_int, default=2048, help="KV token 槽位数")
    parser.add_argument("--temperature", type=float, default=0.6, help="采样温度，0 为贪心解码")
    parser.add_argument("--top-p", type=float, default=0.9, help="随机采样的累计概率阈值")
    parser.add_argument("--repetition-penalty", type=float, default=1.1, help="重复惩罚，1 表示关闭")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--output", type=Path, help="每轮后保存会话记录到 JSON，覆盖指定文件")
    parser.add_argument("--debug", action="store_true", help="显示完整错误堆栈")
    args = parser.parse_args(argv)
    if not math.isfinite(args.temperature) or args.temperature < 0:
        parser.error("--temperature 必须是大于等于 0 的有限数值")
    if not math.isfinite(args.top_p) or not 0 < args.top_p <= 1:
        parser.error("--top-p 必须在 (0, 1] 内")
    if not math.isfinite(args.repetition_penalty) or args.repetition_penalty <= 0:
        parser.error("--repetition-penalty 必须是大于 0 的有限数值")
    if args.max_new_tokens >= min(args.max_seq_len, args.kv_cache_tokens):
        parser.error("--max-new-tokens 必须小于上下文上限和 KV 槽位数，为输入保留空间")
    if not 0 <= args.seed < 2**63:
        parser.error("--seed 必须在 [0, 2**63) 内")
    return args


def prepare_turn(tokenizer, history, user_text, input_budget):
    """Drop complete oldest turns without changing committed history."""
    messages = [*history, {"role": "user", "content": user_text}]
    first_turn = int(messages[0]["role"] == "system")
    removed = 0
    while True:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        count = len(tokenizer.encode(prompt, add_special_tokens=True))
        if count <= input_budget:
            return messages, prompt, removed
        if len(messages) - first_turn <= 1:
            raise ValueError(
                f"本条输入和系统提示共 {count} tokens，可用输入预算为 {input_budget}。"
                "请缩短输入，或调整上下文上限/输出长度。"
            )
        del messages[first_turn:first_turn + 2]
        removed += 1


def save_transcript(path, report):
    if path is None:
        return
    output = path.expanduser().resolve()
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        print(f"记录保存失败：{exc}（可以继续对话）", file=sys.stderr)


def run(args):
    # Keep --help usable without importing CUDA/inference dependencies.
    import torch
    from transformers import AutoTokenizer
    from light_llm.engine.llm import LLM, SamplingParams

    model = args.model.expanduser().resolve()
    if not (model / "config.json").is_file():
        raise ValueError(f"模型目录中没有 config.json：{model}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用；请在服务器的 llm 环境中运行，并检查 CUDA_VISIBLE_DEVICES")
    tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True)
    if not tokenizer.chat_template:
        raise ValueError("此模型没有聊天模板，请使用带 chat_template 的兼容模型目录")
    if "base" in model.name.lower() or model.name in {"Qwen2.5-Math-1.5B", "Qwen2.5-Math-7B"}:
        print(
            "提示：当前选择的是基础模型，带有聊天模板也不代表经过对话训练，可能重复续写。"
            f"终端聊天建议使用 --model {DEFAULT_MODEL}", file=sys.stderr,
        )

    torch.manual_seed(args.seed)
    print(f"light_llm 终端对话\n模型：{model}\n正在加载模型……", flush=True)
    llm = LLM(
        str(model), device="cuda", use_cuda_graph=False,
        max_seq_len=args.max_seq_len, max_gpu_num_blocks=args.kv_cache_tokens,
    )
    tokenizer = llm.tokenizer
    # Qwen base checkpoints may declare only endoftext as EOS. Stop at the
    # chat boundary too, including an attempted start of the next speaker.
    stop_ids = set(llm.stop_token_ids)
    for marker in ("<|im_end|>", "<|im_start|>"):
        if marker in tokenizer.all_special_tokens:
            stop_ids.add(tokenizer.convert_tokens_to_ids(marker))
    llm.stop_token_ids = stop_ids
    params = SamplingParams(
        max_gen_len=args.max_new_tokens, temperature=args.temperature,
        top_p=args.top_p, repetition_penalty=args.repetition_penalty,
    )
    input_budget = min(args.max_seq_len, llm.max_seq_len, args.kv_cache_tokens) - args.max_new_tokens
    if input_budget <= 0:
        raise ValueError("模型实际上下文不足，请减小 --max-new-tokens")
    initial_history = [] if args.system is None else [{"role": "system", "content": args.system}]
    history = list(initial_history)
    report = {
        "model": str(model), "system": args.system,
        "settings": {k: v for k, v in vars(args).items() if k not in {"model", "output", "system"}},
        "events": [],
    }
    print(f"模型已就绪。\n{HELP}", flush=True)
    if args.output:
        print(f"对话记录：{args.output.expanduser().resolve()}")
    while True:
        try:
            user_text = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            break
        if not user_text:
            continue
        command = user_text.lower()
        if command in {"/exit", "/quit"}:
            print("再见。")
            break
        if command == "/help":
            print(HELP)
            continue
        if command == "/clear":
            history = list(initial_history)
            report["events"].append({"type": "clear"})
            save_transcript(args.output, report)
            print("对话历史已清空。")
            continue
        if command.startswith("/"):
            print("未知命令，输入 /help 查看帮助。")
            continue
        try:
            messages, prompt, removed = prepare_turn(tokenizer, history, user_text, input_budget)
        except ValueError as exc:
            print(f"未发送：{exc}")
            continue
        if removed:
            print(f"上下文空间不足，已移除最早的 {removed} 轮对话。")
        print("模型：", end="", flush=True)
        started = time.perf_counter()
        chunks = []
        stream = llm.stream(prompt, params)
        try:
            for deltas in stream:
                chunks.append(deltas[0])
                print(deltas[0], end="", flush=True)
        except KeyboardInterrupt:
            print("\n已取消本轮回复，本轮未加入历史。")
            continue
        finally:
            stream.close()
            # The engine generator has no cancellation cleanup. This demo
            # owns the single request, so release its cache even on Ctrl+C.
            torch.cuda.synchronize()
            llm.model_runner.kv_cache_manager.free_all()
        print(flush=True)
        answer = "".join(chunks)
        history = [*messages, {"role": "assistant", "content": answer}]
        reason = (llm.last_stop_reasons or [None])[0]
        if reason == "length":
            print("（已达到本轮输出长度上限，可继续追问。）")
        report["events"].append({
            "type": "turn", "user": user_text, "assistant": answer,
            "finish_reason": reason, "seconds": time.perf_counter() - started,
            "dropped_history_turns": removed,
        })
        save_transcript(args.output, report)


def main(argv=None):
    args = parse_args(argv)
    try:
        run(args)
    except KeyboardInterrupt:
        print("\n已退出。", file=sys.stderr)
        return 130
    except Exception as exc:
        if args.debug:
            raise
        print(f"运行失败：{exc}\n使用 --debug 查看完整堆栈。", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

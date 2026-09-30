# 终端对话演示

在 63 服务器的 llm 环境中运行：

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate llm
cd /root/ysm/Triton/light_llm-main
CUDA_VISIBLE_DEVICES=0 python examples/run_generation.py
```

模型只加载一次。每次输入一行，回车发送，回复逐步显示；
下一轮会携带之前的问答。例如先输入“解方程 x + 7 = 12”，再追问
“把刚才的答案代回原方程检查一下”。

- `/clear`：清空模型使用的历史，保留系统提示；JSON 记录保留清空前的问答。
- `/exit`、`/quit`：退出；也支持输入结束 EOF。
- `/help`：查看命令。空行不发送，未知的 / 命令会提示错误。
- 回复过程中按 Ctrl+C：取消当前回复，本轮不进入历史，可以重新输入。
- 输入时按 Ctrl+C：退出程序。

## 配置

```bash
# 自定义系统提示，保存本次会话
python examples/run_generation.py \
  --system "你是一位数学助教，请用中文解释解题过程。" \
  --output outputs/chat.json

# 调整输出长度、上下文和采样
python examples/run_generation.py \
  --max-new-tokens 512 --max-seq-len 4096 --kv-cache-tokens 4096 \
  --temperature 0.7 --top-p 0.9

python examples/run_generation.py --help
```

默认加载本地 `/root/ysm/models/Qwen2.5-7B-Instruct`，单卡、关闭 CUDA Graph，
每轮最多输出 256 tokens，上下文与 KV 缓存各 2048 tokens，温度 0.6、top-p 0.9。
该模型已存在于 63 服务器；相比原来的 1.5B 模型，它需要更多显存和加载时间。
脚本使用 tokenizer 的聊天模板，`--system` 可覆盖系统提示。
`--model` 可以指定项目已支持的其他本地模型，该模型必须带聊天模板。
聊天应选择经过指令或对话微调的模型；原来的 Qwen2.5-Math-1.5B 是数学基础模型，
相同聊天输入在 Transformers 中也会重复续写，调高温度无法补上对话训练。
`--temperature 0` 可切换到贪心解码；相同输入重复运行时结果相同是确定性解码的正常现象，
与一条回复内部出现无意义重复不同。

每轮发送前会为输出预留空间，超限时逐轮丢弃最早的完整问答，保留系统提示
和最新输入；最新输入本身过长则拒绝发送，原历史不变。这里的多轮记忆通过
重新拼接消息实现，每轮重新执行 prefill，不跨轮复用 KV 缓存。
`--kv-cache-tokens` 的单位是 token 槽位，不是页。

`--output` 每轮完成或清空后写入 JSON，包含完整问答、清空事件、生成结束原因、
耗时和采样设置，覆盖指定文件；因上下文裁剪而移除的旧问答仍保留在记录中。
耗时包含首次编译及终端输出，不作为性能基准。运行失败时加 `--debug` 查看堆栈。

入口保留为 `examples/run_generation.py`，也可使用 `python -m examples.run_generation`。
原来的 `--prompt` 批量输入和 `--stream` 开关已由默认交互式流式对话替代。
不再使用项目根目录的 `test_smoke.py`。

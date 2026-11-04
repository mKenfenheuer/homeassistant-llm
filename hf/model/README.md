---
license: other
license_name: qwen-research
license_link: LICENSE
language:
  - de
base_model: Qwen/Qwen2.5-3B-Instruct
pipeline_tag: text-generation
tags:
  - gguf
  - home-assistant
  - tool-calling
  - smart-home
  - german
datasets:
  - __DATASET_REPO__
---

# Qwen2.5-3B Home Assistant (German, GGUF)

**Built with Qwen.** LoRA fine-tune of
[Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
that controls a Home Assistant installation in German via tool calls and
answers questions about device states. Quantized to Q4_K_M (1.9 GB) for CPU
inference with llama.cpp.

| File | SHA-256 |
|------|---------|
| `qwen2.5-3b-ha-Q4_K_M.gguf` | `__SHA256__` |

## Usage

The model expects the system prompt and tool schema from
`custom_components/ha_llm_agent/llm_schema.py` of the project repository: a
German instruction, the list of exposed entities (`entity_id | name | area`)
and two tools, `control` (calls Home Assistant services) and `get_state`.
The Home Assistant integration and add-on in the project repository build this
prompt automatically.

```bash
llama-server -m qwen2.5-3b-ha-Q4_K_M.gguf --jinja -c 8192 --port 8080
```

## Training

- Data: [__DATASET_REPO__](https://huggingface.co/datasets/__DATASET_REPO__),
  6,000 synthetic German conversations over randomly generated houses.
- LoRA r=32, alpha=64, all linear layers, 2 epochs, lr 2e-4, loss on
  assistant tokens only. RTX 3090, 3 hours.

## Evaluation

300 held-out conversations in houses not seen during training:

| Model | Correct tool calls | Grounded answers |
|-------|-------------------|------------------|
| Qwen2.5-3B-Instruct (60 samples) | 61.7 % | 50.0 % |
| this model | 98.3 % | 100 % |

End-to-end in Home Assistant 2026.10.4 (demo devices with German names):
23/23 test sentences, median latency 2.8 s on 8 CPU threads.

## Limitations

- German only; supports the services listed in `llm_schema.SERVICES`.
- No timers, automations, weather forecasts or general knowledge questions.
- Above ~250 exposed entities the 8k context is not sufficient.

## License

Derivative of Qwen2.5-3B-Instruct, distributed under the
[Qwen RESEARCH LICENSE AGREEMENT](LICENSE): **non-commercial use only**.
Modifications: LoRA fine-tuning (merged), conversion to GGUF and Q4_K_M
quantization. See `NOTICE`.

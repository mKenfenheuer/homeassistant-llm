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
| `__GGUF__` (current) | `__SHA256__` |
| `qwen2.5-3b-ha-Q4_K_M.gguf` (v1, superseded) | `134b113ee4914735e6d119bc418ca489f04d55acac0d519510fc6896d81a8467` |

## Usage

The model expects the system prompt and tool schema from
`custom_components/ha_llm_agent/llm_schema.py` of the project repository: a
German instruction, the list of exposed entities (`entity_id | name | area`)
and two tools, `control` (calls Home Assistant services) and `get_state`.
The Home Assistant integration and add-on in the project repository build this
prompt automatically.

```bash
llama-server -m __GGUF__ --jinja -c 8192 --port 8080
```

## Training

- Data: [__DATASET_REPO__](https://huggingface.co/datasets/__DATASET_REPO__),
  7,000 synthetic German conversations over randomly generated houses
  (up to 110 entities, group entities, rooms without lights, brand names,
  devices that are not in the list).
- v1: LoRA r=32, alpha=64, all linear layers, 2 epochs, lr 2e-4, loss on
  assistant tokens only. RTX 3090, 3 hours.
- v2: same LoRA setup, 1 further epoch on the v2 data starting from v1,
  lr 1e-4. RTX 3090, 1.7 hours.

## Evaluation

300 held-out conversations (v2 data) in houses not seen during training:

| Model | Correct tool calls | Grounded answers | Unknown device refused |
|-------|-------------------|------------------|------------------------|
| v1 | 89.0 % | 98.4 % | 77.3 % |
| v2 (current) | 94.3 % | 100 % | 100 % |

On the easier v1 test set, v1 reached 98.3 % (base Qwen2.5-3B-Instruct:
61.7 %). On a real house with 107 exposed entities, v2 handles 27 of 31
acceptance sentences correctly (v1: 21).

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

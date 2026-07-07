# CosyVoice API Server

本文档只说明如何启动和运行 `api_server.py`。接口参数和调用示例见
[API Reference](api_reference.md)。

## 启动

基础启动：

```bash
uv run api_server.py \
  --host 0.0.0.0 \
  --port 50001 \
  --model_dir pretrained_models/CosyVoice2-0.5B
```

推荐在 12GB 显存机器上先尝试 FP16：

```bash
uv run api_server.py \
  --host 0.0.0.0 \
  --port 50001 \
  --model_dir pretrained_models/CosyVoice2-0.5B \
  --fp16
```

如果环境已安装并兼容 vLLM，可以进一步开启：

```bash
uv run api_server.py \
  --host 0.0.0.0 \
  --port 50001 \
  --model_dir pretrained_models/CosyVoice2-0.5B \
  --fp16 \
  --load_vllm
```

## 参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--host` | `0.0.0.0` | 监听地址。LAN 调用保持默认即可。 |
| `--port` | `50001` | HTTP 端口。 |
| `--model_dir` | `pretrained_models/CosyVoice2-0.5B` | 本地模型目录或 ModelScope repo id。 |
| `--fp16` | 关闭 | 使用 FP16 推理。需要 CUDA。 |
| `--load_jit` | 关闭 | 加载 JIT artifact。需要模型目录内已有对应文件。 |
| `--load_trt` | 关闭 | 加载或生成 TensorRT artifact。需要 TensorRT 环境。 |
| `--load_vllm` | 关闭 | 使用 vLLM 加速 CosyVoice2/3 的 LLM 部分。 |
| `--trt_concurrent` | `1` | TensorRT 并发 context 数。仅 `--load_trt` 使用。 |
| `--stt_model` | `medium` | faster-whisper 模型名或本地目录。 |
| `--stt_device` | `auto` | STT 设备，可选 `auto`、`cuda`、`cpu`。 |
| `--stt_compute_type` | `int8_float16` | faster-whisper compute type。 |
| `--stt_download_root` | `pretrained_models/stt` | STT 模型下载目录。 |

## STT 模型

默认会加载 faster-whisper `medium`，并把模型文件放在：

```text
pretrained_models/stt
```

如果希望启动时不访问 HuggingFace 检查 revision，可以把 `--stt_model` 指向本地
snapshot 目录：

```bash
uv run api_server.py \
  --host 0.0.0.0 \
  --port 50001 \
  --model_dir pretrained_models/CosyVoice2-0.5B \
  --stt_model pretrained_models/stt/models--Systran--faster-whisper-medium/snapshots/08e178d48790749d25932bbc082711ddcfdfbc4f
```

## 运行说明

- TTS 和 STT 共用一个进程内推理锁，默认同一时刻只跑一个推理任务，避免单卡显存和算力互相抢占。
- `/tts/stream` 返回的是裸 PCM，不是 WAV。调用方需要按 `pcm_s16le`、`24000 Hz`、单声道播放。
- 固定 persona 先调用 `POST /speakers/register` 注册，生成时只传 `zero_shot_spk_id`。
- `/health` 和 `/speakers/exists?spk_id=...` 可以用于服务探活和缓存判断。

## 接口文档

运行后可以访问 FastAPI 自动文档：

```text
http://127.0.0.1:50001/docs
```

项目内维护的接口文档见：

```text
docs/api_reference.md
```

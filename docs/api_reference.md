# CosyVoice API Reference

Base URL 示例：

```text
http://127.0.0.1:50001
```

LAN 调用示例：

```text
http://192.168.5.4:50001
```

## 通用 TTS 参数

`/tts` 和 `/tts/stream` 都使用 `multipart/form-data`。

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `text` | string | 必填 | 要合成的文本。 |
| `mode` | string | `zero_shot` | 可选 `zero_shot`、`cross_lingual`、`sft`、`instruct`、`instruct2`。 |
| `spk_id` | string | 空 | SFT 或 instruct 模式的说话人 id。 |
| `zero_shot_spk_id` | string | 空 | 已注册的 zero-shot persona id。 |
| `instruct_text` | string | 空 | instruct 或 instruct2 的指令文本。 |
| `speed` | float | `1.0` | 语速。流式模式下非 `1.0` 可能不支持。 |
| `text_frontend` | bool | `true` | 是否启用文本前端处理。 |

### 模式参数要求

| mode | 必要参数 |
| --- | --- |
| `zero_shot` | 已注册的 `zero_shot_spk_id`。 |
| `cross_lingual` | 已注册的 `zero_shot_spk_id`。 |
| `sft` | `spk_id`。 |
| `instruct` | `spk_id`、`instruct_text`。 |
| `instruct2` | 已注册的 `zero_shot_spk_id`、`instruct_text`。 |

## GET /health

服务探活。

```bash
curl --noproxy '*' http://127.0.0.1:50001/health
```

返回示例：

```json
{
  "status": "ok"
}
```

## GET /speakers

列出当前进程内已加载的 speaker 和 zero-shot persona。

```bash
curl --noproxy '*' http://127.0.0.1:50001/speakers
```

返回示例：

```json
{
  "speakers": ["kana"],
  "count": 1
}
```

## GET /speakers/exists

检查指定 speaker 或 zero-shot persona 是否已缓存。

```bash
curl --noproxy '*' 'http://127.0.0.1:50001/speakers/exists?spk_id=kana'
```

返回示例：

```json
{
  "spk_id": "kana",
  "exists": true
}
```

## POST /speakers/register

注册 zero-shot persona。生成接口不接收参考音频，必须先通过这个接口注册。

请求使用 `multipart/form-data`。

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `spk_id` | string | 必填 | 要注册的 persona id，例如 `kana`。 |
| `prompt_text` | string | 必填 | 参考音频对应的文本。 |
| `voice` | file | 必填 | 参考音频文件。 |
| `overwrite` | bool | `false` | 已存在时是否覆盖。 |

请求示例：

```bash
curl --noproxy '*' -X POST http://127.0.0.1:50001/speakers/register \
  -F 'spk_id=kana' \
  -F 'prompt_text=希望你以后能够做的比我还好呦。' \
  -F 'voice=@personas/kana/ref.wav'
```

返回示例：

```json
{
  "spk_id": "kana",
  "registered": true,
  "overwritten": false
}
```

如果 `spk_id` 已存在且未传 `overwrite=true`，返回 `409 Conflict`。

## POST /tts

返回完整 WAV 文件。适合不需要边生成边播放的场景。

### zero-shot

```bash
curl --noproxy '*' -X POST http://127.0.0.1:50001/tts \
  -F 'text=你好，这是接口测试。' \
  -F 'mode=zero_shot' \
  -F 'zero_shot_spk_id=kana' \
  --output tts.wav
```

响应：

- `Content-Type: audio/wav`
- body 为完整 WAV 文件。

## POST /tts/stream

返回裸 PCM 流，格式为 signed 16-bit little-endian mono PCM。

响应头：

| Header | 示例 | 说明 |
| --- | --- | --- |
| `Content-Type` | `audio/L16; rate=24000; channels=1` | PCM 音频流。 |
| `X-Sample-Rate` | `24000` | 采样率。 |
| `X-Audio-Format` | `pcm_s16le` | 采样格式。 |

### zero-shot

```bash
curl --noproxy '*' -X POST http://127.0.0.1:50001/tts/stream \
  -F 'text=你好，这是裸音频流测试。' \
  -F 'mode=zero_shot' \
  -F 'zero_shot_spk_id=kana' \
  --output tts.pcm
```

用 ffplay 播放返回的 PCM：

```bash
ffplay -f s16le -ar 24000 -ac 1 tts.pcm
```

## POST /stt

语音转文本，使用 `multipart/form-data`。

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `audio` | file | 必填 | 输入音频。 |
| `language` | string | 空 | 语言代码，例如 `zh`、`en`。为空时自动检测。 |
| `task` | string | `transcribe` | 可选 `transcribe` 或 `translate`。 |
| `initial_prompt` | string | 空 | faster-whisper 初始提示词。 |
| `beam_size` | int | `5` | beam search 大小。 |
| `vad_filter` | bool | `true` | 是否启用 VAD。 |
| `chinese_text` | string | `simplified` | 中文转换，可选 `simplified`、`traditional`、`none`。 |

请求示例：

```bash
curl --noproxy '*' -X POST http://127.0.0.1:50001/stt \
  -F 'audio=@asset/zero_shot_prompt.wav' \
  -F 'language=zh' \
  -F 'chinese_text=simplified'
```

返回示例：

```json
{
  "text": "希望你以后能够做得比我还好呦。",
  "language": "zh",
  "language_probability": 0.99,
  "duration": 3.2,
  "chinese_text": "simplified",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 3.2,
      "text": "希望你以后能够做得比我还好呦。"
    }
  ]
}
```

## 下游推荐流程

固定 persona 的推荐流程：

1. 调 `GET /speakers/exists?spk_id=kana`。
2. 如果 `exists=true`，直接调用 `/tts/stream` 并传 `zero_shot_spk_id=kana`。
3. 如果 `exists=false`，先调用 `POST /speakers/register` 注册 `kana`。
4. 后续请求只传 `zero_shot_spk_id=kana`。

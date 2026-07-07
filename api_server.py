import argparse
import io
import os
import tempfile
import threading
import sys
from pathlib import Path

import torch
import torchaudio
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from modelscope import snapshot_download
from fastapi.responses import Response, StreamingResponse

ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR / "third_party" / "Matcha-TTS"))

from cosyvoice.cli.cosyvoice import AutoModel


parser = argparse.ArgumentParser(
    description="CosyVoice HTTP API server",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
parser.add_argument("--port", type=int, default=50001, help="Port to bind")
parser.add_argument("--model_dir", default="pretrained_models/Fun-CosyVoice3-0.5B", help="Model directory or ModelScope repo id")
parser.add_argument("--fp16", action="store_true", help="Use FP16 inference if supported")
parser.add_argument("--load_jit", action="store_true", help="Load JIT artifacts if available")
parser.add_argument("--load_trt", action="store_true", help="Load TensorRT artifacts if available")
parser.add_argument("--load_vllm", action="store_true", help="Load vLLM artifacts if available")
parser.add_argument("--trt_concurrent", type=int, default=1, help="TensorRT concurrency")
parser.add_argument("--stt_model", default="medium", help="faster-whisper model name or local path")
parser.add_argument("--stt_device", default="auto", choices=["auto", "cuda", "cpu"], help="STT inference device")
parser.add_argument("--stt_compute_type", default="int8_float16", help="faster-whisper compute type")
parser.add_argument(
    "--stt_download_root",
    default="pretrained_models/stt",
    help="Directory for downloaded faster-whisper models",
)
cmd_args = parser.parse_args()

model_dir = Path(cmd_args.model_dir)
if not model_dir.exists():
    model_dir = Path(snapshot_download(cmd_args.model_dir))


def _load_model():
    kwargs = {
        "model_dir": str(model_dir),
        "load_trt": cmd_args.load_trt,
        "fp16": cmd_args.fp16,
        "trt_concurrent": cmd_args.trt_concurrent,
    }
    if (model_dir / "cosyvoice.yaml").exists() or (model_dir / "cosyvoice2.yaml").exists():
        kwargs["load_jit"] = cmd_args.load_jit
    if (model_dir / "cosyvoice2.yaml").exists() or (model_dir / "cosyvoice3.yaml").exists():
        kwargs["load_vllm"] = cmd_args.load_vllm
    return AutoModel(**kwargs)


cosyvoice = _load_model()

app = FastAPI(title="CosyVoice API")
infer_lock = threading.Lock()
stt_model = None
stt_lock = threading.Lock()
opencc_converters = {}
COSYVOICE3_PROMPT_PREFIX = "You are a helpful assistant."
COSYVOICE3_PROMPT_SEPARATOR = "<|endofprompt|>"


def _get_stt_model():
    global stt_model
    if stt_model is not None:
        return stt_model

    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "STT support requires faster-whisper. Install dependencies with: uv sync --locked"
        ) from exc

    device = cmd_args.stt_device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(
        f">> loading STT model: {cmd_args.stt_model} "
        f"(device={device}, compute_type={cmd_args.stt_compute_type}, download_root={cmd_args.stt_download_root})"
    )
    stt_model = WhisperModel(
        cmd_args.stt_model,
        device=device,
        compute_type=cmd_args.stt_compute_type,
        download_root=cmd_args.stt_download_root,
    )
    print(">> STT model loaded.")
    return stt_model


def _convert_chinese_text(text: str, mode: str) -> str:
    if mode == "none" or not text:
        return text
    config = {"simplified": "t2s", "traditional": "s2t"}.get(mode)
    if config is None:
        raise ValueError("chinese_text must be none, simplified, or traditional")

    converter = opencc_converters.get(config)
    if converter is None:
        from opencc import OpenCC

        converter = OpenCC(config)
        opencc_converters[config] = converter
    return converter.convert(text)


def _is_cosyvoice3() -> bool:
    return cosyvoice.__class__.__name__ == "CosyVoice3"


def _cosyvoice3_prompt(text: str) -> str:
    text = text.strip()
    if COSYVOICE3_PROMPT_SEPARATOR in text:
        return text
    return f"{COSYVOICE3_PROMPT_PREFIX}{COSYVOICE3_PROMPT_SEPARATOR}{text}"


with stt_lock:
    _get_stt_model()


def _cleanup(paths: list[str]) -> None:
    for path in paths:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


def _to_int16_pcm(audio: torch.Tensor) -> bytes:
    audio = audio.detach().cpu().flatten().clamp(-1, 1)
    return (audio.numpy() * (2**15)).astype("int16").tobytes()


def _concat_audio(chunks: list[torch.Tensor]) -> torch.Tensor:
    if not chunks:
        return torch.zeros(1, 0)
    normalized = []
    for chunk in chunks:
        chunk = chunk.detach().cpu()
        if chunk.dim() == 1:
            chunk = chunk.unsqueeze(0)
        normalized.append(chunk)
    return torch.cat(normalized, dim=1)


def _wav_bytes(audio: torch.Tensor) -> bytes:
    buffer = io.BytesIO()
    torchaudio.save(buffer, audio, cosyvoice.sample_rate, format="wav")
    return buffer.getvalue()


async def _save_upload(upload: UploadFile, prefix: str) -> str:
    suffix = Path(upload.filename or f"{prefix}.wav").suffix or ".wav"
    fd, path = tempfile.mkstemp(prefix=f"cosyvoice_{prefix}_", suffix=suffix)
    with os.fdopen(fd, "wb") as file:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            file.write(chunk)
    return path


def _validate_generation_request(
    mode: str,
    spk_id: str | None,
    zero_shot_spk_id: str | None,
    instruct_text: str | None,
) -> None:
    if mode == "sft":
        if not spk_id:
            raise ValueError("spk_id is required for sft mode")
        return

    if mode in {"zero_shot", "cross_lingual", "instruct2"}:
        zero_shot_spk_id = zero_shot_spk_id or ""
        if not zero_shot_spk_id:
            raise ValueError(f"zero_shot_spk_id is required for {mode} mode")
        if zero_shot_spk_id not in cosyvoice.frontend.spk2info:
            raise ValueError(f"zero_shot_spk_id {zero_shot_spk_id!r} is not registered")
        if mode == "instruct2" and not instruct_text:
            raise ValueError("instruct_text is required for instruct2 mode")
        return

    if mode == "instruct":
        if not spk_id:
            raise ValueError("spk_id is required for instruct mode")
        if not instruct_text:
            raise ValueError("instruct_text is required for instruct mode")
        return

    raise ValueError("mode must be sft, zero_shot, cross_lingual, instruct, or instruct2")


def _build_model_output(
    mode: str,
    text: str,
    spk_id: str | None,
    zero_shot_spk_id: str | None,
    instruct_text: str | None,
    stream: bool,
    speed: float,
    text_frontend: bool,
):
    if mode == "sft":
        if not spk_id:
            raise ValueError("spk_id is required for sft mode")
        return cosyvoice.inference_sft(text, spk_id, stream=stream, speed=speed, text_frontend=text_frontend)

    if mode == "zero_shot":
        zero_shot_spk_id = zero_shot_spk_id or ""
        if not zero_shot_spk_id:
            raise ValueError("zero_shot_spk_id is required for zero_shot mode")
        if zero_shot_spk_id not in cosyvoice.frontend.spk2info:
            raise ValueError(f"zero_shot_spk_id {zero_shot_spk_id!r} is not registered")
        return cosyvoice.inference_zero_shot(
            text,
            "",
            "",
            zero_shot_spk_id=zero_shot_spk_id,
            stream=stream,
            speed=speed,
            text_frontend=text_frontend,
        )

    if mode == "cross_lingual":
        zero_shot_spk_id = zero_shot_spk_id or ""
        if not zero_shot_spk_id:
            raise ValueError("zero_shot_spk_id is required for cross_lingual mode")
        if zero_shot_spk_id not in cosyvoice.frontend.spk2info:
            raise ValueError(f"zero_shot_spk_id {zero_shot_spk_id!r} is not registered")
        if _is_cosyvoice3():
            text = _cosyvoice3_prompt(text)
        return cosyvoice.inference_cross_lingual(
            text,
            "",
            zero_shot_spk_id=zero_shot_spk_id,
            stream=stream,
            speed=speed,
            text_frontend=text_frontend,
        )

    if mode == "instruct":
        if not spk_id:
            raise ValueError("spk_id is required for instruct mode")
        if not instruct_text:
            raise ValueError("instruct_text is required for instruct mode")
        return cosyvoice.inference_instruct(
            text,
            spk_id,
            instruct_text,
            stream=stream,
            speed=speed,
            text_frontend=text_frontend,
        )

    if mode == "instruct2":
        zero_shot_spk_id = zero_shot_spk_id or ""
        if not zero_shot_spk_id:
            raise ValueError("zero_shot_spk_id is required for instruct2 mode")
        if zero_shot_spk_id not in cosyvoice.frontend.spk2info:
            raise ValueError(f"zero_shot_spk_id {zero_shot_spk_id!r} is not registered")
        if not instruct_text:
            raise ValueError("instruct_text is required for instruct2 mode")
        if _is_cosyvoice3():
            instruct_text = _cosyvoice3_prompt(instruct_text)
        if not hasattr(cosyvoice, "inference_instruct2"):
            raise ValueError("instruct2 mode is not supported by this model")
        return cosyvoice.inference_instruct2(
            text,
            instruct_text,
            "",
            zero_shot_spk_id=zero_shot_spk_id,
            stream=stream,
            speed=speed,
            text_frontend=text_frontend,
        )

    raise ValueError("mode must be sft, zero_shot, cross_lingual, instruct, or instruct2")


def _run_inference(
    mode: str,
    text: str,
    spk_id: str | None,
    zero_shot_spk_id: str | None,
    instruct_text: str | None,
    stream: bool,
    speed: float,
    text_frontend: bool,
):
    model_output = _build_model_output(
        mode=mode,
        text=text,
        spk_id=spk_id,
        zero_shot_spk_id=zero_shot_spk_id,
        instruct_text=instruct_text,
        stream=stream,
        speed=speed,
        text_frontend=text_frontend,
    )
    for chunk in model_output:
        yield chunk["tts_speech"]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/speakers")
def list_speakers():
    speakers = cosyvoice.list_available_spks()
    return {"speakers": speakers, "count": len(speakers)}


@app.get("/speakers/exists")
def speaker_exists(spk_id: str = Query(..., min_length=1)):
    return {"spk_id": spk_id, "exists": spk_id in cosyvoice.frontend.spk2info}


@app.post("/speakers/register")
async def register_speaker(
    spk_id: str = Form(...),
    prompt_text: str = Form(...),
    voice: UploadFile = File(...),
    overwrite: bool = Form(False),
):
    spk_id = spk_id.strip()
    prompt_text = prompt_text.strip()
    if not spk_id:
        raise HTTPException(status_code=400, detail="spk_id must not be empty")
    if not prompt_text:
        raise HTTPException(status_code=400, detail="prompt_text must not be empty")
    if spk_id in cosyvoice.frontend.spk2info and not overwrite:
        raise HTTPException(status_code=409, detail=f"speaker {spk_id!r} already exists")
    if _is_cosyvoice3():
        prompt_text = _cosyvoice3_prompt(prompt_text)

    voice_path = await _save_upload(voice, "voice")
    try:
        with infer_lock:
            cosyvoice.add_zero_shot_spk(prompt_text, voice_path, spk_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        _cleanup([voice_path])

    return {"spk_id": spk_id, "registered": True, "overwritten": overwrite}


@app.post("/stt")
async def transcribe(
    audio: UploadFile = File(...),
    language: str | None = Form(None),
    task: str = Form("transcribe"),
    initial_prompt: str | None = Form(None),
    beam_size: int = Form(5),
    vad_filter: bool = Form(True),
    chinese_text: str = Form("simplified"),
):
    if task not in {"transcribe", "translate"}:
        raise HTTPException(status_code=400, detail="task must be transcribe or translate")
    if chinese_text not in {"none", "simplified", "traditional"}:
        raise HTTPException(status_code=400, detail="chinese_text must be none, simplified, or traditional")

    audio_path = await _save_upload(audio, "stt")

    try:
        # Share the same GPU lock with TTS. This avoids TTS and STT competing for
        # a single card's memory and compute in the default deployment.
        with infer_lock:
            with stt_lock:
                model = _get_stt_model()
            segments_iter, info = model.transcribe(
                audio_path,
                language=language or None,
                task=task,
                initial_prompt=initial_prompt or None,
                beam_size=beam_size,
                vad_filter=vad_filter,
            )
            segments = [
                {
                    "id": index,
                    "start": segment.start,
                    "end": segment.end,
                    "text": _convert_chinese_text(segment.text, chinese_text),
                }
                for index, segment in enumerate(segments_iter)
            ]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        _cleanup([audio_path])

    return {
        "text": "".join(segment["text"] for segment in segments).strip(),
        "language": info.language,
        "language_probability": info.language_probability,
        "duration": info.duration,
        "chinese_text": chinese_text,
        "segments": segments,
    }


@app.post("/tts")
async def synthesize(
    text: str = Form(...),
    mode: str = Form("zero_shot"),
    spk_id: str | None = Form(None),
    zero_shot_spk_id: str | None = Form(None),
    instruct_text: str | None = Form(None),
    speed: float = Form(1.0),
    text_frontend: bool = Form(True),
):
    text = text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text must not be empty")
    try:
        _validate_generation_request(mode, spk_id, zero_shot_spk_id, instruct_text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        with infer_lock:
            chunks = list(
                _run_inference(
                    mode=mode,
                    text=text,
                    spk_id=spk_id,
                    zero_shot_spk_id=zero_shot_spk_id,
                    instruct_text=instruct_text,
                    stream=False,
                    speed=speed,
                    text_frontend=text_frontend,
                )
            )
        output = _wav_bytes(_concat_audio(chunks))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return Response(
        content=output,
        media_type="audio/wav",
        headers={"Content-Disposition": 'attachment; filename="tts.wav"'},
    )


@app.post("/tts/stream")
async def synthesize_pcm_stream(
    text: str = Form(...),
    mode: str = Form("zero_shot"),
    spk_id: str | None = Form(None),
    zero_shot_spk_id: str | None = Form(None),
    instruct_text: str | None = Form(None),
    speed: float = Form(1.0),
    text_frontend: bool = Form(True),
):
    text = text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text must not be empty")
    try:
        _validate_generation_request(mode, spk_id, zero_shot_spk_id, instruct_text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    def generate():
        with infer_lock:
            for chunk in _run_inference(
                mode=mode,
                text=text,
                spk_id=spk_id,
                zero_shot_spk_id=zero_shot_spk_id,
                instruct_text=instruct_text,
                stream=True,
                speed=speed,
                text_frontend=text_frontend,
            ):
                yield _to_int16_pcm(chunk)

    return StreamingResponse(
        generate(),
        media_type=f"audio/L16; rate={cosyvoice.sample_rate}; channels=1",
        headers={"X-Sample-Rate": str(cosyvoice.sample_rate), "X-Audio-Format": "pcm_s16le"},
    )


if __name__ == "__main__":
    uvicorn.run(app, host=cmd_args.host, port=cmd_args.port)

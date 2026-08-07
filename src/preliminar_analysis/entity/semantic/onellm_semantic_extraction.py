from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from semantic_common import (
    build_semantic_prompt,
    build_temporal_windows,
    count_semantic_items,
    discover_video_directories,
    estimate_video_fps,
    list_dense_frames,
    normalize_semantic_output,
    parse_model_json,
    write_csv,
    write_json,
)


DEFAULT_MODEL = "csuhan/OneLLM-7B"


def create_video_clip(frame_paths: tuple[Path, ...], output_path: Path) -> None:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("OpenCV è necessario per creare i clip temporanei.") from error

    first = cv2.imread(str(frame_paths[0]))
    if first is None:
        raise RuntimeError(f"Frame non leggibile: {frame_paths[0]}")
    height, width = first.shape[:2]
    fps = estimate_video_fps(frame_paths)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("Impossibile inizializzare VideoWriter.")

    try:
        for path in frame_paths:
            frame = cv2.imread(str(path))
            if frame is None:
                continue
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))
            writer.write(frame)
    finally:
        writer.release()


class OneLLMInferencer:
    def __init__(
        self,
        model_id: str,
        repository: Path,
        weights_directory: Path,
        device: int,
        dtype: str,
        max_new_tokens: int,
        auto_download: bool,
        master_port: int,
    ) -> None:
        if not repository.exists():
            raise FileNotFoundError(
                "Repository OneLLM non trovata. Esegui: "
                "git clone https://github.com/csuhan/OneLLM "
                f"{repository}"
            )
        sys.path.insert(0, str(repository.resolve()))

        if not weights_directory.exists():
            if not auto_download:
                raise FileNotFoundError(
                    f"Pesi OneLLM assenti: {weights_directory}"
                )
            try:
                from huggingface_hub import snapshot_download
            except ImportError as error:
                raise RuntimeError(
                    "Installa huggingface_hub per scaricare OneLLM-7B."
                ) from error
            print(
                f"Download di {model_id} in {weights_directory}; "
                "il checkpoint occupa circa 15 GB."
            )
            snapshot_download(
                repo_id=model_id,
                local_dir=str(weights_directory),
            )

        checkpoint_path = weights_directory / "consolidated.00-of-01.pth"
        tokenizer_path = weights_directory / "tokenizer.model"
        config_path = repository / "config/llama2/7B.json"
        for required in (checkpoint_path, tokenizer_path, config_path):
            if not required.exists():
                raise FileNotFoundError(f"File OneLLM mancante: {required}")

        try:
            import numpy as np
            import torch
            import torch.distributed as dist
            from fairscale.nn.model_parallel import initialize as fs_init
            from data import video_utils
            from data.conversation_lib import conv_templates
            from model.meta import MetaModel
            from util.misc import default_tensor_type, setup_for_distributed
        except ImportError as error:
            raise RuntimeError(
                "Ambiente OneLLM incompleto. Usa Python 3.9 e installa i "
                "requirements ufficiali del repository."
            ) from error

        self.np = np
        self.torch = torch
        self.dist = dist
        self.video_utils = video_utils
        self.conv_templates = conv_templates
        self.max_new_tokens = max_new_tokens
        self.device = device

        if not torch.cuda.is_available():
            raise RuntimeError("OneLLM richiede una GPU CUDA.")
        torch.cuda.set_device(device)
        torch.manual_seed(1)
        np.random.seed(1)

        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                rank=0,
                world_size=1,
                init_method=f"tcp://127.0.0.1:{master_port}",
            )
        if not fs_init.model_parallel_is_initialized():
            fs_init.initialize_model_parallel(1)
        setup_for_distributed(True)

        if dtype == "bf16" and torch.cuda.is_bf16_supported():
            self.target_dtype = torch.bfloat16
        else:
            self.target_dtype = torch.float16

        print(f"Caricamento di {model_id} su CUDA:{device}...")
        with default_tensor_type(dtype=self.target_dtype, device="cuda"):
            model = MetaModel(
                "onellm",
                str(config_path),
                tokenizer_path=str(tokenizer_path),
            )
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        message = model.load_state_dict(checkpoint, strict=False)
        print(f"Risultato caricamento checkpoint: {message}")
        self.model = model.cuda().eval()

    def __call__(self, video_path: Path, prompt: str) -> str:
        video_features = self.video_utils.load_and_transform_video_data(
            str(video_path),
            str(video_path),
            clip_duration=1,
            clips_per_video=5,
        )
        inputs = video_features[:, :, 0]
        inputs = inputs[None].cuda().to(self.target_dtype)

        conversation = self.conv_templates["v1"].copy()
        conversation.append_message(conversation.roles[0], prompt)
        conversation.append_message(conversation.roles[1], None)
        full_prompt = conversation.get_prompt()

        with self.torch.inference_mode(), self.torch.cuda.amp.autocast(
            dtype=self.target_dtype
        ):
            response = self.model.generate(
                [full_prompt],
                inputs,
                self.max_new_tokens,
                temperature=0.0,
                top_p=1.0,
                modal=["video"],
            )[0]

        if response.startswith(full_prompt):
            response = response[len(full_prompt) :]
        return response.split("###", 1)[0].strip()


def process_video(
    *,
    inferencer: OneLLMInferencer,
    model_name: str,
    video_directory: Path,
    output_directory: Path,
    window_seconds: float,
    stride_seconds: float,
    max_frames: int,
    overwrite: bool,
    keep_raw: bool,
) -> dict[str, Any]:
    output_path = output_directory / f"{video_directory.name}_semantic.json"
    if output_path.exists() and not overwrite:
        return {
            "id_video": video_directory.name,
            "status": "skipped",
            "numero_segmenti": None,
            "numero_elementi": None,
            "numero_errori": None,
            "output": str(output_path),
        }

    frames = list_dense_frames(video_directory)
    if not frames:
        raise FileNotFoundError("Nessun dense frame trovato.")
    windows = build_temporal_windows(
        frames,
        window_seconds=window_seconds,
        stride_seconds=stride_seconds,
        max_frames=max_frames,
    )

    segments = []
    errors = []
    with tempfile.TemporaryDirectory(prefix="onellm_windows_") as temp_dir:
        temporary_directory = Path(temp_dir)
        for window_index, window in enumerate(windows, start=1):
            print(
                f"  [{window_index}/{len(windows)}] {window.segment_id} "
                f"({window.start_time:.3f}-{window.end_time:.3f}s)"
            )
            clip_path = temporary_directory / f"{window.segment_id}.mp4"
            response = ""
            try:
                create_video_clip(window.frame_paths, clip_path)
                prompt = build_semantic_prompt(video_directory.name, window)
                response = inferencer(clip_path, prompt)
                parsed = parse_model_json(response)
                segment = normalize_semantic_output(parsed, window)
                if keep_raw:
                    segment["raw_response"] = response
                segments.append(segment)
            except Exception as error:
                errors.append(
                    {
                        "segment_id": window.segment_id,
                        "start_time": window.start_time,
                        "end_time": window.end_time,
                        "input_frames": window.evidence_frames,
                        "error": f"{type(error).__name__}: {error}",
                        "raw_response": response,
                    }
                )
                print(f"    Errore: {errors[-1]['error']}")

    result = {
        "id_video": video_directory.name,
        "model": model_name,
        "configuration": {
            "window_seconds": window_seconds,
            "stride_seconds": stride_seconds,
            "max_frames_per_window": max_frames,
            "input_type": "temporary_video_from_ordered_dense_frames",
        },
        "numero_segmenti": len(segments),
        "numero_errori": len(errors),
        "segments": segments,
        "errors": errors,
    }
    write_json(output_path, result)
    return {
        "id_video": video_directory.name,
        "status": "completed",
        "numero_segmenti": len(segments),
        "numero_elementi": sum(count_semantic_items(s) for s in segments),
        "numero_errori": len(errors),
        "output": str(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Estrae entità, azioni, eventi e relazioni con OneLLM-7B. "
            "Richiede il repository ufficiale OneLLM."
        )
    )
    parser.add_argument(
        "preprocessing_directory",
        nargs="?",
        default="data/preliminar_analysis/preprocessing",
        type=Path,
    )
    parser.add_argument(
        "output_directory",
        nargs="?",
        default="data/preliminar_analysis/semantic_analysis/onellm",
        type=Path,
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--onellm-repo",
        type=Path,
        default=Path("third_party/OneLLM"),
    )
    parser.add_argument(
        "--weights-directory",
        type=Path,
        default=Path("models/OneLLM-7B"),
    )
    parser.add_argument("--device", type=int, default=4)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--window-seconds", type=float, default=4.0)
    parser.add_argument("--stride-seconds", type=float, default=3.0)
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=900)
    parser.add_argument("--master-port", type=int, default=23872)
    parser.add_argument(
        "--limit-videos",
        type=int,
        default=5,
        help="Numero massimo di video per il pilot; 0 significa tutti.",
    )
    parser.add_argument("--no-auto-download", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-raw", action="store_true")
    args = parser.parse_args()

    video_directories = discover_video_directories(
        args.preprocessing_directory
    )
    if args.limit_videos > 0:
        video_directories = video_directories[: args.limit_videos]
    if not video_directories:
        parser.error("Non sono state trovate cartelle dense_frames.")

    args.output_directory.mkdir(parents=True, exist_ok=True)
    inferencer = OneLLMInferencer(
        model_id=args.model,
        repository=args.onellm_repo,
        weights_directory=args.weights_directory,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        auto_download=not args.no_auto_download,
        master_port=args.master_port,
    )

    rows = []
    for index, video_directory in enumerate(video_directories, start=1):
        print(
            f"[{index}/{len(video_directories)}] "
            f"Analisi OneLLM di {video_directory.name}"
        )
        try:
            rows.append(
                process_video(
                    inferencer=inferencer,
                    model_name=args.model,
                    video_directory=video_directory,
                    output_directory=args.output_directory,
                    window_seconds=args.window_seconds,
                    stride_seconds=args.stride_seconds,
                    max_frames=args.max_frames,
                    overwrite=args.overwrite,
                    keep_raw=args.keep_raw,
                )
            )
        except Exception as error:
            print(f"Errore durante {video_directory.name}: {error}")
            rows.append(
                {
                    "id_video": video_directory.name,
                    "status": "failed",
                    "numero_segmenti": None,
                    "numero_elementi": None,
                    "numero_errori": 1,
                    "output": f"{type(error).__name__}: {error}",
                }
            )

    write_csv(args.output_directory / "riepilogo_video.csv", rows)
    print(f"Risultati salvati in: {args.output_directory}")


if __name__ == "__main__":
    main()

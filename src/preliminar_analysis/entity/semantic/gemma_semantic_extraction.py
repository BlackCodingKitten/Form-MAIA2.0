from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from semantic_common import (
    discover_video_directories,
    process_video_with_inferencer,
    write_csv,
)


DEFAULT_MODEL = "google/gemma-3n-E4B-it"


class GemmaInferencer:
    def __init__(
        self,
        model_id: str,
        device: str,
        dtype: str,
        max_new_tokens: int,
    ) -> None:
        os.environ["CUDA_VISIBLE_DEVICES"] = device

        try:
            import torch
            from PIL import Image
            from transformers import (
                AutoProcessor,
                Gemma3nForConditionalGeneration,
            )
        except ImportError as error:
            raise RuntimeError(
                "Dipendenze Gemma mancanti. Installa transformers>=4.53, "
                "accelerate e Pillow."
            ) from error

        self.torch = torch
        self.Image = Image
        self.max_new_tokens = max_new_tokens

        if dtype == "float16":
            torch_dtype: Any = torch.float16
        elif dtype == "float32":
            torch_dtype = torch.float32
        else:
            torch_dtype = torch.bfloat16

        print(f"Caricamento di {model_id} su CUDA:{device}...")
        self.model = Gemma3nForConditionalGeneration.from_pretrained(
            model_id,
            device_map={"": 0},
            torch_dtype=torch_dtype,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_id)

    def __call__(self, frame_paths: tuple[Path, ...], prompt: str) -> str:
        images = [self.Image.open(path).convert("RGB") for path in frame_paths]
        try:
            content = [
                {"type": "image", "image": image}
                for image in images
            ]
            content.append({"type": "text", "text": prompt})
            messages = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Sei un annotatore scientifico di video. "
                                "Restituisci soltanto JSON valido e verificabile."
                            ),
                        }
                    ],
                },
                {"role": "user", "content": content},
            ]

            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device)
            input_length = inputs["input_ids"].shape[-1]

            with self.torch.inference_mode():
                generation = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )
            generation = generation[0][input_length:]
            return self.processor.decode(
                generation,
                skip_special_tokens=True,
            ).strip()
        finally:
            for image in images:
                image.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Estrae entità, azioni, eventi e relazioni dai dense frame "
            "con Gemma 3n."
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
        default="data/preliminar_analysis/semantic_analysis/gemma",
        type=Path,
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--window-seconds", type=float, default=4.0)
    parser.add_argument("--stride-seconds", type=float, default=3.0)
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=1400)
    parser.add_argument(
        "--limit-videos",
        type=int,
        default=5,
        help="Numero massimo di video per il pilot; 0 significa tutti.",
    )
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
    inferencer = GemmaInferencer(
        model_id=args.model,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
    )

    rows = []
    for index, video_directory in enumerate(video_directories, start=1):
        print(
            f"[{index}/{len(video_directories)}] "
            f"Analisi Gemma di {video_directory.name}"
        )
        try:
            rows.append(
                process_video_with_inferencer(
                    model_name=args.model,
                    video_directory=video_directory,
                    output_directory=args.output_directory,
                    window_seconds=args.window_seconds,
                    stride_seconds=args.stride_seconds,
                    max_frames=args.max_frames,
                    inferencer=inferencer,
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

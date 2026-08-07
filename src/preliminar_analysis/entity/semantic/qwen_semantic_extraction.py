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


DEFAULT_MODEL = "Qwen/Qwen2.5-Omni-3B"


class QwenInferencer:
    def __init__(
        self,
        model_id: str,
        device: str,
        dtype: str,
        max_new_tokens: int,
        flash_attention: bool,
    ) -> None:
        os.environ["CUDA_VISIBLE_DEVICES"] = device

        try:
            import torch
            from qwen_omni_utils import process_mm_info
            from transformers import (
                Qwen2_5OmniProcessor,
                Qwen2_5OmniThinkerForConditionalGeneration,
            )
        except ImportError as error:
            raise RuntimeError(
                "Dipendenze Qwen mancanti. Installa transformers, accelerate "
                "e qwen-omni-utils[decord]."
            ) from error

        self.torch = torch
        self.process_mm_info = process_mm_info
        self.max_new_tokens = max_new_tokens

        if dtype == "bfloat16":
            torch_dtype: Any = torch.bfloat16
        elif dtype == "float16":
            torch_dtype = torch.float16
        else:
            torch_dtype = "auto"

        model_kwargs: dict[str, Any] = {
            "torch_dtype": torch_dtype,
            "device_map": {"": 0},
        }
        if flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        print(f"Caricamento di {model_id} su CUDA:{device}...")
        self.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            model_id,
            **model_kwargs,
        ).eval()
        self.processor = Qwen2_5OmniProcessor.from_pretrained(model_id)

    def __call__(self, frame_paths: tuple[Path, ...], prompt: str) -> str:
        content = [
            {"type": "image", "image": str(path.resolve())}
            for path in frame_paths
        ]
        content.append({"type": "text", "text": prompt})

        conversation = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Sei un annotatore scientifico di video. Produci solo "
                            "JSON verificabile e non aggiungere fatti non osservati."
                        ),
                    }
                ],
            },
            {"role": "user", "content": content},
        ]

        text = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )
        audios, images, videos = self.process_mm_info(
            conversation,
            use_audio_in_video=False,
        )
        inputs = self.processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=False,
        )
        inputs = inputs.to(self.model.device)

        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                use_audio_in_video=False,
            )

        input_ids = inputs.get("input_ids")
        if (
            input_ids is not None
            and generated.shape[1] > input_ids.shape[1]
            and self.torch.equal(
                generated[:, : input_ids.shape[1]],
                input_ids,
            )
        ):
            generated = generated[:, input_ids.shape[1] :]

        return self.processor.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Estrae entità, azioni, eventi e relazioni dai dense frame "
            "con Qwen2.5-Omni."
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
        default="data/preliminar_analysis/semantic_analysis/qwen",
        type=Path,
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="3")
    parser.add_argument(
        "--dtype",
        choices=["auto", "bfloat16", "float16"],
        default="auto",
    )
    parser.add_argument("--window-seconds", type=float, default=4.0)
    parser.add_argument("--stride-seconds", type=float, default=3.0)
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=16384)
    parser.add_argument(
        "--limit-videos",
        type=int,
        default=5,
        help="Numero massimo di video per il pilot; 0 significa tutti.",
    )
    parser.add_argument("--flash-attention", action="store_true")
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
    inferencer = QwenInferencer(
        model_id=args.model,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        flash_attention=args.flash_attention,
    )

    rows = []
    for index, video_directory in enumerate(video_directories, start=1):
        print(
            f"[{index}/{len(video_directories)}] "
            f"Analisi Qwen di {video_directory.name}"
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

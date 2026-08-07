from __future__ import annotations

import os

# ==============================================================================
# FORZATURA GPU: Isola le GPU fisiche 4, 5, 6 e 7 PRIMA di importare torch.
# PyTorch le mapperà come cuda:0, cuda:1, cuda:2, cuda:3 logiche.
# ==============================================================================
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

import argparse
from pathlib import Path

from semantic_common import (
    discover_video_directories,
    process_video_with_inferencer,
    write_csv,
)

DEFAULT_MODEL = "google/gemma-3-27b-it"


class Gemma27Inferencer:
    def __init__(
        self,
        model_id: str,
        max_new_tokens: int,
    ) -> None:
        try:
            import torch
            from PIL import Image
            from transformers import (
                AutoProcessor,
                Gemma3ForConditionalGeneration,
            )
        except ImportError as error:
            raise RuntimeError(
                "Dipendenze mancanti. Installa transformers, accelerate e Pillow."
            ) from error

        self.torch = torch
        self.Image = Image
        self.max_new_tokens = max_new_tokens

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA non disponibile.")

        gpu_count = torch.cuda.device_count()
        print(f"Caricamento di {model_id} in BF16 (device_map='auto')...")
        print(f"GPU visibili a PyTorch: {gpu_count} (Corrispondenti a GPU fisiche 4, 5, 6, 7)")

        # Caricamento con ripartizione automatica dei layer sulle GPU visibili
        self.model = (
            Gemma3ForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
            .eval()
        )

        self.processor = AutoProcessor.from_pretrained(model_id)

        # Il device primario (dove risiede il primo layer per gli input)
        try:
            self.input_device = self.model.device
        except AttributeError:
            self.input_device = torch.device("cuda:0")

        print("\nDevice map del modello Gemma-27B:")
        if hasattr(self.model, "hf_device_map"):
            for module_name, device in self.model.hf_device_map.items():
                print(f"  {module_name or '<root>'}: {device}")
        print()

    def __call__(
        self,
        frame_paths: tuple[Path, ...],
        prompt: str,
    ) -> str:
        images = [
            self.Image.open(path).convert("RGB")
            for path in frame_paths
        ]

        try:
            content = [
                {
                    "type": "image",
                    "image": image,
                }
                for image in images
            ]

            content.append(
                {
                    "type": "text",
                    "text": prompt,
                }
            )

            messages = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Sei un annotatore scientifico di video. "
                                "Restituisci esclusivamente JSON valido, "
                                "conciso e verificabile."
                            ),
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": content,
                },
            ]

            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )

            # Spostamento degli input sul device primario del modello
            inputs = {
                key: (
                    value.to(self.input_device)
                    if hasattr(value, "to")
                    else value
                )
                for key, value in inputs.items()
            }

            if "pixel_values" in inputs:
                inputs["pixel_values"] = (
                    inputs["pixel_values"].to(dtype=self.torch.bfloat16)
                )

            input_length = inputs["input_ids"].shape[-1]

            with self.torch.inference_mode():
                generation = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )

            generation = generation[0, input_length:]

            text = self.processor.decode(
                generation,
                skip_special_tokens=True,
            ).strip()

            return text

        finally:
            for image in images:
                image.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Estrae entità, azioni, eventi e relazioni "
            "con Gemma-3-27B in locale sulle GPU 4, 5, 6 e 7."
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
        default="data/preliminar_analysis/entity/entity_semantic/gemma-27B",
        type=Path,
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
    )

    parser.add_argument(
        "--window-seconds",
        type=float,
        default=4.0,
    )

    parser.add_argument(
        "--stride-seconds",
        type=float,
        default=3.0,
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=8192,
    )

    parser.add_argument(
        "--limit-videos",
        type=int,
        default=0,
        help="Numero massimo di video da analizzare; 0 = tutti.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    parser.add_argument(
        "--keep-raw",
        action="store_true",
    )

    args = parser.parse_args()

    video_directories = discover_video_directories(
        args.preprocessing_directory
    )

    if args.limit_videos > 0:
        video_directories = video_directories[: args.limit_videos]

    if not video_directories:
        parser.error("Non sono state trovate cartelle dense_frames.")

    args.output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    inferencer = Gemma27Inferencer(
        model_id=args.model,
        max_new_tokens=args.max_new_tokens,
    )

    rows = []

    for index, video_directory in enumerate(video_directories, start=1):
        print(
            f"[{index}/{len(video_directories)}] "
            f"Analisi Gemma-3-27B di {video_directory.name}",
            flush=True,
        )

        try:
            row = process_video_with_inferencer(
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
            rows.append(row)

        except Exception as error:
            print(
                f"Errore durante {video_directory.name}: "
                f"{type(error).__name__}: {error}",
                flush=True,
            )
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

    write_csv(
        args.output_directory / "riepilogo_video.csv",
        rows,
    )

    print(f"Risultati salvati in: {args.output_directory}")


if __name__ == "__main__":
    main()
from __future__ import annotations

import argparse
import os
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
        devices: str,
        max_new_tokens: int,
        max_memory_gib: int,
    ) -> None:
        # Esempio: --devices 1,2
        # All'interno del processo queste GPU diventano cuda:0 e cuda:1.
        os.environ["CUDA_VISIBLE_DEVICES"] = devices

        try:
            import torch
            from PIL import Image
            from transformers import (
                AutoProcessor,
                BitsAndBytesConfig,
                Gemma3ForConditionalGeneration,
            )
        except ImportError as error:
            raise RuntimeError(
                "Dipendenze mancanti. Installa transformers, accelerate, "
                "bitsandbytes e Pillow."
            ) from error

        self.torch = torch
        self.Image = Image
        self.max_new_tokens = max_new_tokens
        self.devices = [item.strip() for item in devices.split(",") if item.strip()]

        if not self.devices:
            raise ValueError("Specifica almeno una GPU con --devices.")

        if torch.cuda.device_count() != len(self.devices):
            raise RuntimeError(
                f"GPU richieste: {self.devices}; GPU visibili a PyTorch: "
                f"{torch.cuda.device_count()}."
            )

        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        # Le GPU fisiche 1 e 2 sono rimappate internamente a 0 e 1.
        max_memory = {
            index: f"{max_memory_gib}GiB"
            for index in range(len(self.devices))
        }

        print(
            f"Caricamento di {model_id} in 4-bit sulle GPU fisiche "
            f"{','.join(self.devices)}..."
        )
        print(f"Limite memoria per GPU: {max_memory_gib} GiB")

        # "balanced" distribuisce i layer sulle GPU visibili invece di
        # concentrare il modello sulla prima GPU.
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            quantization_config=quantization,
            device_map="balanced",
            max_memory=max_memory,
        ).eval()

        self.processor = AutoProcessor.from_pretrained(model_id)

        # Gli input entrano dalla prima GPU visibile; Accelerate trasferisce
        # poi automaticamente gli stati tra i layer shardati.
        self.input_device = torch.device("cuda:0")

        print("Device map:")
        for module_name, device in self.model.hf_device_map.items():
            print(f"  {module_name or '<root>'}: {device}")

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
                                "Restituisci esclusivamente JSON valido, "
                                "conciso e verificabile."
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
            )

            inputs = {
                key: value.to(self.input_device)
                if hasattr(value, "to")
                else value
                for key, value in inputs.items()
            }

            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(
                    dtype=self.torch.bfloat16
                )

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
            "con google/gemma-3-27b-it distribuito su più GPU."
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

    parser.add_argument("--model", default=DEFAULT_MODEL)

    parser.add_argument(
        "--devices",
        default="1,2",
        help="GPU fisiche separate da virgola. Default: 1,2.",
    )

    parser.add_argument(
        "--max-memory-gib",
        type=int,
        default=40,
        help="Memoria massima utilizzabile per ciascuna GPU visibile.",
    )

    parser.add_argument("--window-seconds", type=float, default=4.0)
    parser.add_argument("--stride-seconds", type=float, default=3.0)
    parser.add_argument("--max-frames", type=int, default=8)

    parser.add_argument("--max-new-tokens", type=int, default=3000)

    parser.add_argument(
        "--limit-videos",
        type=int,
        default=0,
        help="Numero massimo di video da analizzare; 0 = tutti.",
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

    inferencer = Gemma27Inferencer(
        model_id=args.model,
        devices=args.devices,
        max_new_tokens=args.max_new_tokens,
        max_memory_gib=args.max_memory_gib,
    )

    rows = []

    for index, video_directory in enumerate(video_directories, start=1):
        print(
            f"[{index}/{len(video_directories)}] "
            f"Analisi Gemma-3-27B di {video_directory.name}"
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

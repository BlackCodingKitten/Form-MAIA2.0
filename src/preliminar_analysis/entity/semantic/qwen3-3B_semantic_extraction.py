from __future__ import annotations

import argparse
from pathlib import Path

from semantic_common import (
    discover_video_directories,
    process_video_with_inferencer,
    write_csv,
)

MODEL_ID = "Qwen/Qwen3-VL-30B-A3B-Instruct"
GPU_IDS = (5, 6)


def _device_is_allowed(device) -> bool:
    if isinstance(device, int):
        return device in GPU_IDS
    text = str(device)
    return text in {"5", "6", "cuda:5", "cuda:6"}


class Qwen30Inferencer:
    def __init__(
        self,
        model_id: str,
        max_new_tokens: int,
        flash_attention: bool,
    ) -> None:
        try:
            import torch
            from PIL import Image
            from transformers import (
                AutoProcessor,
                Qwen3VLMoeForConditionalGeneration,
            )
        except ImportError as error:
            raise RuntimeError(
                "Dipendenze Qwen3-VL mancanti. Installa una versione recente "
                "di transformers, accelerate, torch e Pillow."
            ) from error

        self.torch = torch
        self.Image = Image
        self.max_new_tokens = max_new_tokens
        self.input_device = torch.device("cuda:5")

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA non disponibile.")

        if torch.cuda.device_count() <= max(GPU_IDS):
            raise RuntimeError(
                f"Servono almeno {max(GPU_IDS) + 1} GPU visibili a PyTorch "
                f"per usare cuda:5 e cuda:6. GPU visibili: "
                f"{torch.cuda.device_count()}."
            )

        # Solo cuda:5 e cuda:6 possono contenere i pesi.
        # cuda:5 ha un po' più di margine libero perché riceve anche input/output.
        max_memory = {
            5: "38GiB",
            6: "40GiB",
            "cpu": "1MiB",
        }

        load_kwargs = {
            "dtype": torch.bfloat16,
            "device_map": "balanced",
            "max_memory": max_memory,
        }

        if flash_attention:
            load_kwargs["attn_implementation"] = "flash_attention_2"

        print(
            f"Caricamento di {model_id} in BF16 su cuda:5 e cuda:6..."
        )

        self.model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_id,
            **load_kwargs,
        ).eval()

        self.processor = AutoProcessor.from_pretrained(model_id)

        # Controllo forte: nessun layer deve finire su altre GPU/CPU/disk.
        invalid_devices = {
            str(device)
            for device in self.model.hf_device_map.values()
            if not _device_is_allowed(device)
        }

        if invalid_devices:
            raise RuntimeError(
                "Il modello non è stato confinato a cuda:5 e cuda:6. "
                f"Device inattesi: {sorted(invalid_devices)}"
            )

        print("Device map Qwen30:")
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
                {
                    "role": "user",
                    "content": content,
                },
            ]

            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )

            # Gli input entrano da cuda:5. Accelerate sposta automaticamente
            # gli hidden state tra i layer distribuiti su cuda:5 e cuda:6.
            for key, value in list(inputs.items()):
                if hasattr(value, "to"):
                    value = value.to(self.input_device)
                    if key in {"pixel_values", "pixel_values_videos"}:
                        value = value.to(dtype=self.torch.bfloat16)
                    inputs[key] = value

            input_length = inputs["input_ids"].shape[-1]

            with self.torch.inference_mode():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )

            generated_ids = generated_ids[:, input_length:]

            return self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()

        finally:
            for image in images:
                image.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analisi semantica dei dense frame con "
            "Qwen3-VL-30B-A3B-Instruct su cuda:5 e cuda:6."
        )
    )

    parser.add_argument(
        "preprocessing_directory",
        nargs="?",
        type=Path,
        default="data/preliminar_analysis/preprocessing",
        help=(
            "Directory contenente una cartella per video, ognuna con "
            "dense_frames/."
        ),
    )

    parser.add_argument(
        "output_directory",
        nargs="?",
        type=Path,
        default="data/preliminar_analysis/entity/entity_semantic/qwen-30B",
    )

    parser.add_argument("--model", default=MODEL_ID)

    parser.add_argument("--window-seconds", type=float, default=4.0)
    parser.add_argument("--stride-seconds", type=float, default=3.0)
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=3000)

    parser.add_argument(
        "--limit-videos",
        type=int,
        default=0,
        help="Numero massimo di video; 0 = tutti.",
    )

    parser.add_argument(
        "--flash-attention",
        action="store_true",
        help="Usa FlashAttention2 se installata.",
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
        parser.error(
            "Non sono state trovate cartelle video contenenti dense_frames."
        )

    args.output_directory.mkdir(parents=True, exist_ok=True)

    inferencer = Qwen30Inferencer(
        model_id=args.model,
        max_new_tokens=args.max_new_tokens,
        flash_attention=args.flash_attention,
    )

    rows = []

    for index, video_directory in enumerate(video_directories, start=1):
        print(
            f"[{index}/{len(video_directories)}] "
            f"Qwen30 semantic: {video_directory.name}"
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
            print(
                f"Errore durante {video_directory.name}: "
                f"{type(error).__name__}: {error}"
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

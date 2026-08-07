#!/usr/bin/env python3

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import argparse
import json
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

import cv2
import numpy as np


MODEL_ID = "csuhan/OneLLM-7B"
MODEL_NAME = "OneLLM-7B"
GPU_FISICA = 4
OUTPUT_FILE = "eventi_onellm_av.json"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
ENTITY_FIELDS = ("agente", "oggetto", "strumento", "origine", "destinazione", "luogo")
EVENT_TYPES = {
    "azione",
    "movimento",
    "manipolazione",
    "interazione",
    "comunicazione",
    "evento_sonoro",
    "comparsa",
    "scomparsa",
    "cambiamento_di_stato",
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def extract_json(text):
    text = text.replace("```json", "").replace("```", "").strip()
    return json.loads(text[text.find("{"):text.rfind("}") + 1])


def find_video(directory, video_id):
    return next(
        (
            path
            for path in directory.rglob("*")
            if path.is_file()
            and path.suffix.lower() in VIDEO_EXTENSIONS
            and path.stem == video_id
        ),
        None,
    )


def frame_path(frame, preprocessing_directory, video_id, segment_id):
    path = Path(frame["file"])
    return path if path.exists() else (
        preprocessing_directory
        / video_id
        / "segment_frames"
        / segment_id
        / path.name
    )


def select_frames(frames, number):
    return [
        frames[index]
        for index in np.linspace(
            0,
            len(frames) - 1,
            min(number, len(frames)),
            dtype=int,
        )
    ]


def annotate_frames(
    frames,
    entities,
    preprocessing_directory,
    video_id,
    segment_id,
    output_directory,
):
    annotated = []

    for index, frame in enumerate(frames, 1):
        image = cv2.imread(
            str(frame_path(frame, preprocessing_directory, video_id, segment_id))
        )

        if image is None:
            continue

        frame_id = f"frame_{index:02d}"
        timestamp = float(frame["timestamp"])
        cv2.putText(
            image,
            f"{frame_id} | {timestamp:.3f} s",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2,
        )

        for entity in entities:
            observation = min(
                entity.get("osservazioni", []),
                key=lambda item: abs(item["timestamp"] - timestamp),
                default=None,
            )

            if not observation or abs(observation["timestamp"] - timestamp) > 0.30:
                continue

            x1, y1, x2, y2 = map(int, observation["riquadro_xyxy"])
            label = f'{entity["id_entita"]}: {entity["classe_detector"]}'
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                image,
                label,
                (x1, max(55, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )

        path = output_directory / f"{frame_id}.jpg"
        cv2.imwrite(str(path), image)
        annotated.append({
            "id_frame": frame_id,
            "timestamp": timestamp,
            "path": str(path),
        })

    return annotated


def run(command):
    return subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )


def create_silent_audio(path, duration):
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\x00\x00" * max(1, int(duration * 16000)))


def prepare_media(
    source_video,
    segment,
    manifest_segment,
    preprocessing_directory,
    video_id,
    output_directory,
    number_of_frames,
):
    duration = max(0.1, segment["fine"] - segment["inizio"])
    frames = annotate_frames(
        select_frames(manifest_segment["frames"], number_of_frames),
        segment["entita_visibili"],
        preprocessing_directory,
        video_id,
        segment["id_segmento"],
        output_directory,
    )

    if not frames:
        raise RuntimeError("Nessun frame del segmento è stato letto.")

    silent_video = output_directory / "video_senza_audio.mp4"
    audiovisual_video = output_directory / "segmento_audiovisivo.mp4"
    audio_path = output_directory / "segmento_audio.wav"
    fps = len(frames) / duration

    run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-framerate", str(fps),
        "-start_number", "1",
        "-i", str(output_directory / "frame_%02d.jpg"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-r", str(fps),
        str(silent_video),
    ]).check_returncode()

    audio_result = run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", str(segment["inizio"]),
        "-t", str(duration),
        "-i", str(source_video),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_f32le",
        str(audio_path),
    ])
    audio_present = audio_result.returncode == 0 and audio_path.exists()

    if not audio_present:
        create_silent_audio(audio_path, duration)

    mux_result = run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(silent_video),
        "-ss", str(segment["inizio"]),
        "-t", str(duration),
        "-i", str(source_video),
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-c:v", "copy",
        "-c:a", "aac",
        "-ac", "1",
        "-ar", "16000",
        "-shortest",
        str(audiovisual_video),
    ])

    if mux_result.returncode != 0:
        shutil.copy2(silent_video, audiovisual_video)

    return frames, audiovisual_video, audio_path, audio_present


def build_prompt(segment, frames):
    entities = "\n".join(
        f'- {entity["id_entita"]}: {entity["classe_detector"]}'
        for entity in segment["entita_visibili"]
    ) or "- nessuna entità tracciata"
    frame_list = "\n".join(
        f'- {frame["id_frame"]}: {frame["timestamp"]:.3f} secondi'
        for frame in frames
    )

    return f"""
Analizza congiuntamente la sequenza video e il suo audio.

Segmento: {segment["id_segmento"]}
Intervallo originale: {segment["inizio"]:.3f}-{segment["fine"]:.3f} secondi

Frame annotati:
{frame_list}

Entità tracciate:
{entities}

Estrai soltanto eventi direttamente osservabili o udibili. Non inferire
intenzioni, emozioni, cause invisibili o conseguenze future. Usa esclusivamente
gli ID elencati per le entità tracciate. Gli oggetti visibili ma non tracciati
devono essere descritti in "entita_non_tracciate". Usa verbi all'infinito.

Per un evento visivo indica almeno un frame di supporto. Un evento esclusivamente
sonoro può avere "frame_di_supporto": [] e deve contenere "audio" in
"modalita_evidenza".

Restituisci esclusivamente questo JSON:
{{
  "eventi": [
    {{
      "predicato": "verbo all'infinito",
      "tipo_evento": "azione|movimento|manipolazione|interazione|comunicazione|evento_sonoro|comparsa|scomparsa|cambiamento_di_stato",
      "descrizione": "descrizione breve in italiano",
      "agente": [],
      "oggetto": [],
      "strumento": [],
      "origine": [],
      "destinazione": [],
      "luogo": [],
      "entita_non_tracciate": [],
      "modalita_evidenza": ["video", "audio"],
      "frame_di_supporto": [],
      "evidenza_audio": "",
      "stato_precedente": "",
      "stato_successivo": "",
      "confidenza": 0.0
    }}
  ]
}}
""".strip()


def normalize_events(data, segment, frames):
    entity_ids = {
        entity["id_entita"]
        for entity in segment["entita_visibili"]
    }
    frame_times = {
        frame["id_frame"]: frame["timestamp"]
        for frame in frames
    }
    events = []

    for index, event in enumerate(data.get("eventi", []), 1):
        modalities = [
            modality
            for modality in event.get("modalita_evidenza", [])
            if modality in {"video", "audio"}
        ]
        support = [
            frame_id
            for frame_id in event.get("frame_di_supporto", [])
            if frame_id in frame_times
        ]

        if not support and "audio" not in modalities:
            continue

        try:
            confidence = float(event.get("confidenza", 0))
        except (TypeError, ValueError):
            confidence = 0.0

        normalized = {
            "id_evento": f'{segment["id_segmento"]}_evento_{index:03d}',
            "predicato": str(event.get("predicato", "")).strip().lower(),
            "tipo_evento": event.get("tipo_evento", "azione"),
            "descrizione": str(event.get("descrizione", "")).strip(),
            **{
                field: [
                    entity_id
                    for entity_id in event.get(field, [])
                    if entity_id in entity_ids
                ]
                for field in ENTITY_FIELDS
            },
            "entita_non_tracciate": list(event.get("entita_non_tracciate", [])),
            "modalita_evidenza": modalities,
            "frame_di_supporto": support,
            "evidenza_audio": str(event.get("evidenza_audio", "")).strip(),
            "inizio": (
                min(frame_times[frame_id] for frame_id in support)
                if support
                else segment["inizio"]
            ),
            "fine": (
                max(frame_times[frame_id] for frame_id in support)
                if support
                else segment["fine"]
            ),
            "stato_precedente": str(event.get("stato_precedente", "")).strip(),
            "stato_successivo": str(event.get("stato_successivo", "")).strip(),
            "confidenza": min(1.0, max(0.0, confidence)),
        }
        normalized["tipo_evento"] = (
            normalized["tipo_evento"]
            if normalized["tipo_evento"] in EVENT_TYPES
            else "azione"
        )
        events.append(normalized)

    return events


import sys

import torch


def load_model(args):
    repository = args.onellm_repository.resolve()
    sys.path.insert(0, str(repository))

    import torch.distributed as dist
    from fairscale.nn.model_parallel import initialize as fs_init
    from data import video_utils
    from data.conversation_lib import conv_templates
    from data.fintune_dataset import make_audio_features
    from model.meta import MetaModel
    from util.misc import default_tensor_type, setup_for_distributed

    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            rank=0,
            world_size=1,
            init_method=f"tcp://127.0.0.1:{args.master_port}",
        )

    fs_init.initialize_model_parallel(1)
    torch.cuda.set_device(0)
    setup_for_distributed(True)

    with default_tensor_type(dtype=torch.float16, device="cuda"):
        model = MetaModel(
            "onellm",
            str(args.llama_config),
            tokenizer_path=str(args.tokenizer_path),
        )

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint.get("model", checkpoint), strict=False)
    model.cuda().eval()

    def load_video(path):
        return video_utils.load_and_transform_video_data(
            str(path),
            str(path),
            clip_duration=1,
            clips_per_video=5,
        )[:, :, 0]

    def load_audio(path):
        return make_audio_features(
            str(path),
            mel_bins=128,
        ).transpose(0, 1)[None]

    return model, conv_templates, load_video, load_audio


def generate(model, conv_templates, prompt, inputs, modality, max_new_tokens):
    conversation = conv_templates["v1"].copy()
    conversation.append_message(conversation.roles[0], prompt)
    conversation.append_message(conversation.roles[1], None)
    full_prompt = conversation.get_prompt()

    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.float16):
        response = model.generate(
            [full_prompt],
            None if inputs is None else inputs[None].cuda().half(),
            max_new_tokens,
            temperature=0,
            top_p=1,
            modal=[modality],
        )[0]

    return response[len(full_prompt):].split("###")[0].strip()


def run_model(
    model_data,
    video_path,
    audio_path,
    audio_present,
    frames,
    prompt,
    max_new_tokens,
):
    model, conv_templates, load_video, load_audio = model_data
    video_analysis = generate(
        model,
        conv_templates,
        (
            "Analizza soltanto la componente video del segmento. "
            "Elenca in italiano azioni, movimenti, interazioni e cambiamenti "
            "di stato visibili, usando gli ID scritti nei fotogrammi."
        ),
        load_video(video_path),
        "video",
        max_new_tokens // 2,
    )
    audio_analysis = generate(
        model,
        conv_templates,
        (
            "Analizza soltanto la componente audio del segmento. "
            "Elenca in italiano parlato, suoni, rumori ed eventi udibili."
        ),
        load_audio(audio_path),
        "audio",
        max_new_tokens // 2,
    )
    fusion_prompt = f"""
Osservazioni video:
{video_analysis}

Osservazioni audio:
{audio_analysis}

{prompt}

Integra le due analisi e restituisci esclusivamente il JSON richiesto.
""".strip()
    response = generate(
        model,
        conv_templates,
        fusion_prompt,
        None,
        "image",
        max_new_tokens,
    )

    return response, {
        "integrazione_modalita": (
            "analisi video e audio separate con OneLLM, "
            "seguite da fusione testuale nello stesso modello"
        ),
        "analisi_video_grezza": video_analysis,
        "analisi_audio_grezza": audio_analysis,
    }


def main():
    parser = argparse.ArgumentParser(
        description=f"Estrae eventi audiovisivi con {MODEL_NAME}."
    )
    parser.add_argument("entita_segmenti_json", type=Path)
    parser.add_argument("preprocessing_directory", type=Path)
    parser.add_argument("video_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=1000)
    parser.add_argument("--onellm-repository", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--llama-config", type=Path, required=True)
    parser.add_argument("--master-port", type=int, default=29544)
    args = parser.parse_args()

    args.output_directory.mkdir(parents=True, exist_ok=True)
    output_path = args.output_directory / OUTPUT_FILE
    source = read_json(args.entita_segmenti_json)
    output = read_json(output_path) if output_path.exists() else {
        "modello": args.model,
        "gpu_fisica": f"CUDA:{GPU_FISICA}",
        "video": [],
    }
    processed = {
        (video["id_video"], segment["id_segmento"])
        for video in output["video"]
        for segment in video["segmenti"]
    }
    model_data = load_model(args)

    for video_index, video in enumerate(source["video"], 1):
        video_id = video["id_video"]
        source_video = find_video(args.video_directory, video_id)

        if source_video is None:
            print(f"Video originale non trovato: {video_id}")
            continue

        manifest = {
            segment["segment_id"]: segment
            for segment in read_json(
                args.preprocessing_directory / video_id / "segments.json"
            )["segments"]
        }
        video_output = next(
            (item for item in output["video"] if item["id_video"] == video_id),
            None,
        )

        if video_output is None:
            video_output = {"id_video": video_id, "segmenti": []}
            output["video"].append(video_output)

        for segment_index, segment in enumerate(video["segmenti"], 1):
            key = video_id, segment["id_segmento"]

            if key in processed:
                continue

            print(
                f'[{video_index}/{len(source["video"])}] {video_id} - '
                f'[{segment_index}/{len(video["segmenti"])}] '
                f'{segment["id_segmento"]} su CUDA:{GPU_FISICA}'
            )
            result = {
                "id_segmento": segment["id_segmento"],
                "inizio": segment["inizio"],
                "fine": segment["fine"],
                "eventi": [],
            }

            try:
                with tempfile.TemporaryDirectory() as temporary_directory:
                    frames, video_path, audio_path, audio_present = prepare_media(
                        source_video,
                        segment,
                        manifest[segment["id_segmento"]],
                        args.preprocessing_directory,
                        video_id,
                        Path(temporary_directory),
                        args.frames,
                    )
                    response, details = run_model(
                        model_data,
                        video_path,
                        audio_path,
                        audio_present,
                        frames,
                        build_prompt(segment, frames),
                        args.max_new_tokens,
                    )

                result["audio_originale_presente"] = audio_present
                result.update(details)
                result["risposta_grezza"] = response
                result["eventi"] = normalize_events(
                    extract_json(response),
                    segment,
                    frames,
                )

            except Exception as error:
                result["errore"] = str(error)

            video_output["segmenti"].append(result)
            write_json(output_path, output)

    print(f"Creato: {output_path}")


if __name__ == "__main__":
    main()

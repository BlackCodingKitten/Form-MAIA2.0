import argparse
import json
from pathlib import Path


def load_json(path):
    """Load a JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path, data):
    """Save a JSON file."""
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def valid_track(track, min_observations, min_confidence, min_duration):
    """Check whether a track satisfies all filters."""
    return (
        track["numero_osservazioni"] >= min_observations
        and track["confidenza_media"] >= min_confidence
        and track["durata_traccia"] >= min_duration
    )


def main():
    parser = argparse.ArgumentParser(
        description="Crea due JSON complessivi con entità e associazioni entità-segmenti."
    )
    parser.add_argument("tracking_directory", type=Path)
    parser.add_argument("preprocessing_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--minimum-observations", type=int, default=3)
    parser.add_argument("--minimum-confidence", type=float, default=0.30)
    parser.add_argument("--minimum-duration", type=float, default=0.50)
    args = parser.parse_args()

    args.output_directory.mkdir(parents=True, exist_ok=True)
    tracking_files = sorted(args.tracking_directory.glob("*_tracciamento.json"))

    if not tracking_files:
        parser.error("Nessun file *_tracciamento.json trovato.")

    all_entities = []
    all_segments = []

    for index, tracking_file in enumerate(tracking_files, 1):
        print(f"[{index}/{len(tracking_files)}] Elaborazione di {tracking_file.name}")

        try:
            tracking = load_json(tracking_file)
            video_id = tracking["id_video"]
            segments = load_json(
                args.preprocessing_directory / video_id / "segments.json"
            )["segments"]

            accepted_tracks = [
                track
                for track in tracking["tracce"]
                if valid_track(
                    track,
                    args.minimum_observations,
                    args.minimum_confidence,
                    args.minimum_duration,
                )
            ]

            entities = [
                {
                    "id_entita": f"entita_{i:03d}",
                    "id_traccia": track["id_traccia"],
                    "id_classe": track["id_classe"],
                    "classe_detector": track["nome_classe"],
                    "etichetta_semantica": None,
                    "prima_comparsa": track["prima_comparsa"],
                    "ultima_comparsa": track["ultima_comparsa"],
                    "durata_visibile": track["durata_traccia"],
                    "numero_osservazioni": track["numero_osservazioni"],
                    "confidenza_media": track["confidenza_media"],
                    "osservazioni": track["osservazioni"],
                }
                for i, track in enumerate(accepted_tracks, 1)
            ]

            mapped_segments = []

            for segment in segments:
                start, end = segment["start_time"], segment["end_time"]

                visible_entities = []

                for entity in entities:
                    observations = [
                        observation
                        for observation in entity["osservazioni"]
                        if start <= observation["timestamp"] <= end
                    ]

                    if observations:
                        visible_entities.append({
                            "id_entita": entity["id_entita"],
                            "id_traccia": entity["id_traccia"],
                            "classe_detector": entity["classe_detector"],
                            "etichetta_semantica": entity["etichetta_semantica"],
                            "prima_osservazione_segmento": observations[0]["timestamp"],
                            "ultima_osservazione_segmento": observations[-1]["timestamp"],
                            "numero_osservazioni_segmento": len(observations),
                            "osservazioni": observations,
                        })

                mapped_segments.append({
                    "id_segmento": segment["segment_id"],
                    "inizio": start,
                    "fine": end,
                    "durata": segment["duration"],
                    "inquadrature_origine": segment.get("source_shots", []),
                    "numero_entita_visibili": len(visible_entities),
                    "entita_visibili": visible_entities,
                })

            all_entities.append({
                "id_video": video_id,
                "numero_tracce_originali": len(tracking["tracce"]),
                "numero_entita_accettate": len(entities),
                "numero_tracce_escluse": len(tracking["tracce"]) - len(entities),
                "entita": entities,
            })

            all_segments.append({
                "id_video": video_id,
                "numero_segmenti": len(mapped_segments),
                "segmenti": mapped_segments,
            })

        except Exception as error:
            print(f"Errore durante l'elaborazione di {tracking_file.name}: {error}")

    save_json(
        args.output_directory / "entita.json",
        {"numero_video": len(all_entities), "video": all_entities},
    )
    save_json(
        args.output_directory / "entita_segmenti.json",
        {"numero_video": len(all_segments), "video": all_segments},
    )

    print(f"Creati: {args.output_directory / 'entita.json'}")
    print(f"Creati: {args.output_directory / 'entita_segmenti.json'}")


if __name__ == "__main__":
    main()

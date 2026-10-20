"""Format saved detector tracks without loading detection models."""


def format_track_summary(summary: dict, style: str = "compact") -> str:
    label = summary.get("label", "unknown")
    track_id = summary.get("track_id", "unknown")
    movement = summary.get("movement", "unknown")
    start_pos = summary.get("start_pos", "unknown")
    end_pos = summary.get("end_pos", "unknown")
    relations = summary.get("relations", []) or []

    if style == "narrative":
        movement_phrase = movement.replace(",", "")
        base = f"{label} #{track_id} is {movement_phrase} to {end_pos}"
        if start_pos and start_pos != "unknown":
            base = f"{label} #{track_id} is {movement_phrase} from {start_pos} to {end_pos}"
        relation_phrases = [f"{label} #{track_id} is {rel}" for rel in relations]
        if relation_phrases:
            return "; ".join([base] + relation_phrases)
        return base

    base = f"{label} #{track_id}: {start_pos} -> {end_pos}, {movement}"
    if relations:
        base += f"; relations: {', '.join(relations)}"
    return base


def format_track_summaries(summaries: list, style: str = "compact") -> list:
    return [format_track_summary(s, style=style) for s in summaries]

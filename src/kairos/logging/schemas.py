def scene_schema(
    scene_index: int,
    start_seconds: float,
    end_seconds: float,
    start_timecode: str,
    end_timecode: str,
) -> dict:
    """Build the standard Kairos scene dictionary used downstream."""
    return {
        "scene_index": scene_index,
        "start_timecode": start_timecode,
        "end_timecode": end_timecode,
        "start_seconds": start_seconds,
        "end_seconds": end_seconds,
        "duration_seconds": end_seconds - start_seconds,
    }


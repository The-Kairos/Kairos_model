# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 09:38:21 UTC | MRkoyixoWPc_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 131.115 | 0.660 | 36.798 | 16.169 | 19.618 | 24.527 | 2.079 |

## 2026-06-25 09:38:21 UTC | MRkoyixoWPc_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/MRkoyixoWPc_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `131.115` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.660 |
| save_clips | - |
| sample_frames | 0.450 |
| caption_frames | 20.750 |
| sample_fps | 1.785 |
| detect_object_yolo | 6.756 |
| audio_scan | 11.823 |
| asr_timings | 11.785 |
| ast_timings | 13.181 |
| describe_scenes | 16.169 |
| summarize_scenes | 19.618 |
| synthesize_synopsis | 24.527 |
| make_embedding | 2.079 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 21.205 |
| branch_yolo_total | 8.546 |
| branch_audio_total | 36.798 |

# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 19:50:14 UTC | BlZaRZqMZ84_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 91.480 | 0.684 | 32.240 | 6.192 | 4.786 | 22.201 | 1.874 |

## 2026-06-24 19:50:14 UTC | BlZaRZqMZ84_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/BlZaRZqMZ84_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `91.480` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.684 |
| save_clips | - |
| sample_frames | 0.246 |
| caption_frames | 13.667 |
| sample_fps | 1.680 |
| detect_object_yolo | 6.507 |
| audio_scan | 10.775 |
| asr_timings | 11.663 |
| ast_timings | 9.793 |
| describe_scenes | 6.192 |
| summarize_scenes | 4.786 |
| synthesize_synopsis | 22.201 |
| make_embedding | 1.874 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 13.919 |
| branch_yolo_total | 8.192 |
| branch_audio_total | 32.240 |

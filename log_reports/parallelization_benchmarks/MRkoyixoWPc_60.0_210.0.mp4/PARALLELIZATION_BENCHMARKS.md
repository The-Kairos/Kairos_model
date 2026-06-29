# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 09:43:54 UTC | MRkoyixoWPc_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 204.481 | 0.663 | 56.786 | 30.908 | 29.300 | 26.748 | 3.653 |

## 2026-06-25 09:43:54 UTC | MRkoyixoWPc_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/MRkoyixoWPc_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `204.481` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.663 |
| save_clips | - |
| sample_frames | 1.066 |
| caption_frames | 42.864 |
| sample_fps | 2.151 |
| detect_object_yolo | 8.906 |
| audio_scan | 16.036 |
| asr_timings | 10.704 |
| ast_timings | 30.037 |
| describe_scenes | 30.908 |
| summarize_scenes | 29.300 |
| synthesize_synopsis | 26.748 |
| make_embedding | 3.653 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 43.936 |
| branch_yolo_total | 11.063 |
| branch_audio_total | 56.786 |

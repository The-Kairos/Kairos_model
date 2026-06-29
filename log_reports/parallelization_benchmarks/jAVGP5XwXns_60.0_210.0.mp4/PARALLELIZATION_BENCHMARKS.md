# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 10:25:57 UTC | jAVGP5XwXns_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 182.254 | 0.645 | 59.214 | 18.181 | 20.639 | 23.227 | 3.847 |

## 2026-06-26 10:25:57 UTC | jAVGP5XwXns_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/jAVGP5XwXns_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `182.254` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.645 |
| save_clips | - |
| sample_frames | 1.122 |
| caption_frames | 43.067 |
| sample_fps | 2.242 |
| detect_object_yolo | 8.678 |
| audio_scan | 13.882 |
| asr_timings | 12.241 |
| ast_timings | 33.083 |
| describe_scenes | 18.181 |
| summarize_scenes | 20.639 |
| synthesize_synopsis | 23.227 |
| make_embedding | 3.847 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 44.195 |
| branch_yolo_total | 10.926 |
| branch_audio_total | 59.214 |

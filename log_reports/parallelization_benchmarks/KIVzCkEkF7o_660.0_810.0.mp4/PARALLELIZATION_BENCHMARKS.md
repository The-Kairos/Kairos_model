# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 06:28:40 UTC | KIVzCkEkF7o_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 192.809 | 0.774 | 56.704 | 18.312 | 21.567 | 28.143 | 4.208 |

## 2026-06-25 06:28:40 UTC | KIVzCkEkF7o_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/KIVzCkEkF7o_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `192.809` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.774 |
| save_clips | - |
| sample_frames | 1.363 |
| caption_frames | 48.102 |
| sample_fps | 2.399 |
| detect_object_yolo | 9.838 |
| audio_scan | 14.831 |
| asr_timings | 7.420 |
| ast_timings | 34.445 |
| describe_scenes | 18.312 |
| summarize_scenes | 21.567 |
| synthesize_synopsis | 28.143 |
| make_embedding | 4.208 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 49.470 |
| branch_yolo_total | 12.242 |
| branch_audio_total | 56.704 |

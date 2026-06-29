# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 05:17:14 UTC | gS7RgLgSw-o_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 178.161 | 0.810 | 94.092 | 15.065 | 7.596 | 11.406 | 3.029 |

## 2026-06-26 05:17:14 UTC | gS7RgLgSw-o_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/gS7RgLgSw-o_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `178.161` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.810 |
| save_clips | - |
| sample_frames | 0.905 |
| caption_frames | 33.408 |
| sample_fps | 2.219 |
| detect_object_yolo | 8.187 |
| audio_scan | 10.939 |
| asr_timings | 58.163 |
| ast_timings | 24.981 |
| describe_scenes | 15.065 |
| summarize_scenes | 7.596 |
| synthesize_synopsis | 11.406 |
| make_embedding | 3.029 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 34.319 |
| branch_yolo_total | 10.411 |
| branch_audio_total | 94.092 |

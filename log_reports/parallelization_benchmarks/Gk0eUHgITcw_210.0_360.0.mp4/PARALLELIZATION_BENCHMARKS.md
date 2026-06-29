# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 02:06:09 UTC | Gk0eUHgITcw_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 138.553 | 0.629 | 50.958 | 11.383 | 11.472 | 11.686 | 3.291 |

## 2026-06-25 02:06:09 UTC | Gk0eUHgITcw_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/Gk0eUHgITcw_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `138.553` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.629 |
| save_clips | - |
| sample_frames | 0.930 |
| caption_frames | 36.207 |
| sample_fps | 2.010 |
| detect_object_yolo | 8.574 |
| audio_scan | 14.898 |
| asr_timings | 9.185 |
| ast_timings | 26.866 |
| describe_scenes | 11.383 |
| summarize_scenes | 11.472 |
| synthesize_synopsis | 11.686 |
| make_embedding | 3.291 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 37.142 |
| branch_yolo_total | 10.589 |
| branch_audio_total | 50.958 |

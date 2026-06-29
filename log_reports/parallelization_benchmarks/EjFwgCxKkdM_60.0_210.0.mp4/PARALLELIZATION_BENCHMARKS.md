# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 23:46:12 UTC | EjFwgCxKkdM_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 168.901 | 0.700 | 59.404 | 13.506 | 14.749 | 8.629 | 4.636 |

## 2026-06-24 23:46:12 UTC | EjFwgCxKkdM_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/EjFwgCxKkdM_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `168.901` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.700 |
| save_clips | - |
| sample_frames | 1.543 |
| caption_frames | 51.817 |
| sample_fps | 2.347 |
| detect_object_yolo | 10.152 |
| audio_scan | 12.821 |
| asr_timings | 8.202 |
| ast_timings | 38.372 |
| describe_scenes | 13.506 |
| summarize_scenes | 14.749 |
| synthesize_synopsis | 8.629 |
| make_embedding | 4.636 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 53.366 |
| branch_yolo_total | 12.506 |
| branch_audio_total | 59.404 |

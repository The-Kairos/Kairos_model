# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 23:40:33 UTC | EjFwgCxKkdM_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 154.640 | 0.729 | 52.142 | 13.462 | 16.184 | 15.138 | 3.541 |

## 2026-06-24 23:40:33 UTC | EjFwgCxKkdM_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/EjFwgCxKkdM_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `154.640` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.729 |
| save_clips | - |
| sample_frames | 1.083 |
| caption_frames | 39.703 |
| sample_fps | 2.178 |
| detect_object_yolo | 9.029 |
| audio_scan | 13.956 |
| asr_timings | 8.165 |
| ast_timings | 30.012 |
| describe_scenes | 13.462 |
| summarize_scenes | 16.184 |
| synthesize_synopsis | 15.138 |
| make_embedding | 3.541 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 40.792 |
| branch_yolo_total | 11.213 |
| branch_audio_total | 52.142 |

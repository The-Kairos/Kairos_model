# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 18:44:59 UTC | AIYfAAX7XL0_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 126.616 | 0.679 | 45.446 | 12.012 | 8.656 | 14.725 | 2.820 |

## 2026-06-24 18:44:59 UTC | AIYfAAX7XL0_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/AIYfAAX7XL0_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `126.616` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.679 |
| save_clips | - |
| sample_frames | 0.916 |
| caption_frames | 30.005 |
| sample_fps | 2.004 |
| detect_object_yolo | 7.948 |
| audio_scan | 14.958 |
| asr_timings | 9.488 |
| ast_timings | 20.991 |
| describe_scenes | 12.012 |
| summarize_scenes | 8.656 |
| synthesize_synopsis | 14.725 |
| make_embedding | 2.820 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 30.927 |
| branch_yolo_total | 9.959 |
| branch_audio_total | 45.446 |

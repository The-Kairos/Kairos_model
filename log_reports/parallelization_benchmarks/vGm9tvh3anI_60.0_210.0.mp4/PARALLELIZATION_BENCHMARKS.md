# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 02:09:36 UTC | vGm9tvh3anI_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 137.657 | 0.651 | 51.889 | 12.697 | 5.867 | 11.389 | 3.518 |

## 2026-06-27 02:09:36 UTC | vGm9tvh3anI_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/vGm9tvh3anI_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `137.657` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.651 |
| save_clips | - |
| sample_frames | 0.903 |
| caption_frames | 38.369 |
| sample_fps | 2.069 |
| detect_object_yolo | 8.905 |
| audio_scan | 11.835 |
| asr_timings | 10.574 |
| ast_timings | 29.472 |
| describe_scenes | 12.697 |
| summarize_scenes | 5.867 |
| synthesize_synopsis | 11.389 |
| make_embedding | 3.518 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 39.278 |
| branch_yolo_total | 10.980 |
| branch_audio_total | 51.889 |

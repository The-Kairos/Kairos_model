# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 01:37:48 UTC | boJssjt0HGk_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 150.565 | 0.688 | 56.670 | 9.686 | 8.177 | 10.798 | 4.198 |

## 2026-06-26 01:37:48 UTC | boJssjt0HGk_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/boJssjt0HGk_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `150.565` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.688 |
| save_clips | - |
| sample_frames | 1.187 |
| caption_frames | 45.887 |
| sample_fps | 2.181 |
| detect_object_yolo | 9.697 |
| audio_scan | 12.943 |
| asr_timings | 7.553 |
| ast_timings | 36.165 |
| describe_scenes | 9.686 |
| summarize_scenes | 8.177 |
| synthesize_synopsis | 10.798 |
| make_embedding | 4.198 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 47.080 |
| branch_yolo_total | 11.884 |
| branch_audio_total | 56.670 |

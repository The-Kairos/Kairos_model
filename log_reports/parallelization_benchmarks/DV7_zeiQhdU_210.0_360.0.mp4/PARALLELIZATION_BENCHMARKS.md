# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 22:54:00 UTC | DV7_zeiQhdU_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 132.513 | 0.769 | 49.314 | 14.184 | 5.754 | 9.770 | 3.241 |

## 2026-06-24 22:54:00 UTC | DV7_zeiQhdU_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/DV7_zeiQhdU_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `132.513` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.769 |
| save_clips | - |
| sample_frames | 0.829 |
| caption_frames | 36.318 |
| sample_fps | 2.131 |
| detect_object_yolo | 8.807 |
| audio_scan | 11.792 |
| asr_timings | 10.500 |
| ast_timings | 27.013 |
| describe_scenes | 14.184 |
| summarize_scenes | 5.754 |
| synthesize_synopsis | 9.770 |
| make_embedding | 3.241 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 37.153 |
| branch_yolo_total | 10.944 |
| branch_audio_total | 49.314 |

# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 13:48:20 UTC | kM_8DQ-iJcU_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 155.734 | 0.696 | 50.503 | 17.286 | 11.911 | 25.233 | 3.372 |

## 2026-06-26 13:48:20 UTC | kM_8DQ-iJcU_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/kM_8DQ-iJcU_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `155.734` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.696 |
| save_clips | - |
| sample_frames | 0.954 |
| caption_frames | 34.165 |
| sample_fps | 2.053 |
| detect_object_yolo | 8.109 |
| audio_scan | 5.446 |
| asr_timings | 17.268 |
| ast_timings | 27.780 |
| describe_scenes | 17.286 |
| summarize_scenes | 11.911 |
| synthesize_synopsis | 25.233 |
| make_embedding | 3.372 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 35.125 |
| branch_yolo_total | 10.168 |
| branch_audio_total | 50.503 |

# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 09:50:13 UTC | MlXof8hF4ck_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 218.783 | 0.762 | 62.256 | 17.750 | 32.549 | 40.532 | 7.009 |

## 2026-06-25 09:50:13 UTC | MlXof8hF4ck_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/MlXof8hF4ck_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `218.783` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.762 |
| save_clips | - |
| sample_frames | 1.046 |
| caption_frames | 44.178 |
| sample_fps | 2.218 |
| detect_object_yolo | 9.043 |
| audio_scan | 13.818 |
| asr_timings | 18.566 |
| ast_timings | 29.862 |
| describe_scenes | 17.750 |
| summarize_scenes | 32.549 |
| synthesize_synopsis | 40.532 |
| make_embedding | 7.009 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 45.230 |
| branch_yolo_total | 11.266 |
| branch_audio_total | 62.256 |

# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 17:35:14 UTC | Sl0DKauc_oU_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 188.350 | 0.841 | 67.132 | 14.588 | 18.469 | 13.119 | 5.059 |

## 2026-06-25 17:35:14 UTC | Sl0DKauc_oU_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/Sl0DKauc_oU_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `188.350` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.841 |
| save_clips | - |
| sample_frames | 1.438 |
| caption_frames | 53.039 |
| sample_fps | 2.546 |
| detect_object_yolo | 10.718 |
| audio_scan | 14.978 |
| asr_timings | 11.205 |
| ast_timings | 40.941 |
| describe_scenes | 14.588 |
| summarize_scenes | 18.469 |
| synthesize_synopsis | 13.119 |
| make_embedding | 5.059 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 54.483 |
| branch_yolo_total | 13.270 |
| branch_audio_total | 67.132 |

# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-28 08:58:53 UTC | qiM0OMBRY30_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 133.309 | 0.798 | 59.241 | 7.757 | 6.070 | 8.232 | 3.021 |

## 2026-06-28 08:58:53 UTC | qiM0OMBRY30_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/qiM0OMBRY30_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `133.309` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.798 |
| save_clips | - |
| sample_frames | 1.008 |
| caption_frames | 35.431 |
| sample_fps | 2.187 |
| detect_object_yolo | 8.163 |
| audio_scan | 12.850 |
| asr_timings | 21.961 |
| ast_timings | 24.422 |
| describe_scenes | 7.757 |
| summarize_scenes | 6.070 |
| synthesize_synopsis | 8.232 |
| make_embedding | 3.021 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 36.445 |
| branch_yolo_total | 10.356 |
| branch_audio_total | 59.241 |

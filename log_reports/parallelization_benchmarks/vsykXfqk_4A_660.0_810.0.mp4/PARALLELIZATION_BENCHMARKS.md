# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 02:46:13 UTC | vsykXfqk_4A_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 126.344 | 0.774 | 48.910 | 10.156 | 7.197 | 8.854 | 3.263 |

## 2026-06-27 02:46:13 UTC | vsykXfqk_4A_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/vsykXfqk_4A_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `126.344` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.774 |
| save_clips | - |
| sample_frames | 0.889 |
| caption_frames | 34.444 |
| sample_fps | 2.183 |
| detect_object_yolo | 8.248 |
| audio_scan | 14.029 |
| asr_timings | 8.838 |
| ast_timings | 26.034 |
| describe_scenes | 10.156 |
| summarize_scenes | 7.197 |
| synthesize_synopsis | 8.854 |
| make_embedding | 3.263 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 35.339 |
| branch_yolo_total | 10.437 |
| branch_audio_total | 48.910 |

# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 02:05:02 UTC | vGm9tvh3anI_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 122.291 | 0.659 | 46.092 | 8.313 | 9.775 | 11.937 | 2.767 |

## 2026-06-27 02:05:02 UTC | vGm9tvh3anI_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/vGm9tvh3anI_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `122.291` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.659 |
| save_clips | - |
| sample_frames | 0.770 |
| caption_frames | 30.827 |
| sample_fps | 1.980 |
| detect_object_yolo | 7.778 |
| audio_scan | 11.811 |
| asr_timings | 12.999 |
| ast_timings | 21.273 |
| describe_scenes | 8.313 |
| summarize_scenes | 9.775 |
| synthesize_synopsis | 11.937 |
| make_embedding | 2.767 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 31.603 |
| branch_yolo_total | 9.764 |
| branch_audio_total | 46.092 |

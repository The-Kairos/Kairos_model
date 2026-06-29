# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 16:37:41 UTC | 86fm_miRHWI_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 139.916 | 0.522 | 46.815 | 14.245 | 12.943 | 13.311 | 3.896 |

## 2026-06-24 16:37:41 UTC | 86fm_miRHWI_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/86fm_miRHWI_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `139.916` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.522 |
| save_clips | - |
| sample_frames | 0.772 |
| caption_frames | 46.037 |
| sample_fps | 0.914 |
| detect_object_yolo | 9.082 |
| audio_scan | 3.762 |
| asr_timings | 0.000 |
| ast_timings | 33.042 |
| describe_scenes | 14.245 |
| summarize_scenes | 12.943 |
| synthesize_synopsis | 13.311 |
| make_embedding | 3.896 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 46.815 |
| branch_yolo_total | 10.001 |
| branch_audio_total | 36.812 |

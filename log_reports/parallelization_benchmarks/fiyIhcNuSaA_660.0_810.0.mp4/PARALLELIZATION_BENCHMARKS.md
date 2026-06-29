# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 04:39:57 UTC | fiyIhcNuSaA_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 79.199 | 0.781 | 32.875 | 4.087 | 4.433 | 15.857 | 1.826 |

## 2026-06-26 04:39:57 UTC | fiyIhcNuSaA_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/fiyIhcNuSaA_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `79.199` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.781 |
| save_clips | - |
| sample_frames | 0.164 |
| caption_frames | 9.886 |
| sample_fps | 1.806 |
| detect_object_yolo | 6.098 |
| audio_scan | 16.197 |
| asr_timings | 9.397 |
| ast_timings | 7.273 |
| describe_scenes | 4.087 |
| summarize_scenes | 4.433 |
| synthesize_synopsis | 15.857 |
| make_embedding | 1.826 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 10.056 |
| branch_yolo_total | 7.909 |
| branch_audio_total | 32.875 |

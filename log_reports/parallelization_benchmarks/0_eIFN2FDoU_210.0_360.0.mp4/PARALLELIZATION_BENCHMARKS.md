# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-21 20:53:49 UTC | 0_eIFN2FDoU_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 0.060 | - | - | - | - | - | - |
| 2026-06-22 12:31:55 UTC | 0_eIFN2FDoU_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 79.586 | 0.751 | 26.785 | 8.188 | 8.389 | 17.018 | 1.324 |

## 2026-06-21 20:53:49 UTC | 0_eIFN2FDoU_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/0_eIFN2FDoU_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `0.060` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | - |
| save_clips | - |
| sample_frames | - |
| caption_frames | - |
| sample_fps | - |
| detect_object_yolo | - |
| audio_scan | - |
| asr_timings | - |
| ast_timings | - |
| describe_scenes | - |
| summarize_scenes | - |
| synthesize_synopsis | - |
| make_embedding | - |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | - |
| branch_yolo_total | - |
| branch_audio_total | - |

## 2026-06-22 12:31:55 UTC | 0_eIFN2FDoU_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/0_eIFN2FDoU_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `79.586` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.751 |
| save_clips | - |
| sample_frames | 0.094 |
| caption_frames | 8.226 |
| sample_fps | 1.740 |
| detect_object_yolo | 5.706 |
| audio_scan | 10.691 |
| asr_timings | 11.724 |
| ast_timings | 4.361 |
| describe_scenes | 8.188 |
| summarize_scenes | 8.389 |
| synthesize_synopsis | 17.018 |
| make_embedding | 1.324 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 8.326 |
| branch_yolo_total | 7.452 |
| branch_audio_total | 26.785 |

# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 18:37:17 UTC | AEf_3sgs_Ak_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 108.911 | 0.770 | 38.422 | 8.267 | 6.886 | 18.691 | 2.279 |

## 2026-06-24 18:37:17 UTC | AEf_3sgs_Ak_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/AEf_3sgs_Ak_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `108.911` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.770 |
| save_clips | - |
| sample_frames | 0.572 |
| caption_frames | 22.811 |
| sample_fps | 1.974 |
| detect_object_yolo | 6.826 |
| audio_scan | 12.903 |
| asr_timings | 9.644 |
| ast_timings | 15.866 |
| describe_scenes | 8.267 |
| summarize_scenes | 6.886 |
| synthesize_synopsis | 18.691 |
| make_embedding | 2.279 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 23.389 |
| branch_yolo_total | 8.806 |
| branch_audio_total | 38.422 |

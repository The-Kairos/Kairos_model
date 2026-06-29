# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-21 09:41:32 UTC | 2X46BBkcCeY_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 175.581 | 2.039 | 66.400 | 13.255 | 6.723 | 6.164 | 5.143 |
| 2026-06-21 21:20:30 UTC | 2X46BBkcCeY_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 183.393 | 2.046 | 63.949 | 11.997 | 16.465 | 6.157 | 5.063 |

## 2026-06-21 09:41:32 UTC | 2X46BBkcCeY_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/2X46BBkcCeY_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `175.581` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 2.039 |
| save_clips | - |
| sample_frames | 3.971 |
| caption_frames | 53.310 |
| sample_fps | 6.780 |
| detect_object_yolo | 10.467 |
| audio_scan | 13.830 |
| asr_timings | 11.964 |
| ast_timings | 40.597 |
| describe_scenes | 13.255 |
| summarize_scenes | 6.723 |
| synthesize_synopsis | 6.164 |
| make_embedding | 5.143 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 57.287 |
| branch_yolo_total | 17.253 |
| branch_audio_total | 66.400 |

## 2026-06-21 21:20:30 UTC | 2X46BBkcCeY_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/2X46BBkcCeY_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `183.393` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 2.046 |
| save_clips | - |
| sample_frames | 4.000 |
| caption_frames | 54.561 |
| sample_fps | 6.910 |
| detect_object_yolo | 10.840 |
| audio_scan | 14.008 |
| asr_timings | 9.095 |
| ast_timings | 40.838 |
| describe_scenes | 11.997 |
| summarize_scenes | 16.465 |
| synthesize_synopsis | 6.157 |
| make_embedding | 5.063 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 58.568 |
| branch_yolo_total | 17.757 |
| branch_audio_total | 63.949 |

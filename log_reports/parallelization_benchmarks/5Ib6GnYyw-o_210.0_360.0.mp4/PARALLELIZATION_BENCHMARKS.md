# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 17:39:37 UTC | 5Ib6GnYyw-o_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 156.436 | 0.852 | 55.136 | 17.645 | 9.538 | 9.795 | 3.918 |
| 2026-06-24 11:32:08 UTC | 5Ib6GnYyw-o_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 169.557 | 0.869 | 55.077 | 18.417 | 13.065 | 18.095 | 3.863 |

## 2026-06-23 17:39:37 UTC | 5Ib6GnYyw-o_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/5Ib6GnYyw-o_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `156.436` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.852 |
| save_clips | - |
| sample_frames | 1.613 |
| caption_frames | 44.988 |
| sample_fps | 2.511 |
| detect_object_yolo | 9.063 |
| audio_scan | 14.866 |
| asr_timings | 8.370 |
| ast_timings | 31.891 |
| describe_scenes | 17.645 |
| summarize_scenes | 9.538 |
| synthesize_synopsis | 9.795 |
| make_embedding | 3.918 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 46.607 |
| branch_yolo_total | 11.580 |
| branch_audio_total | 55.136 |

## 2026-06-24 11:32:08 UTC | 5Ib6GnYyw-o_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/5Ib6GnYyw-o_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `169.557` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.869 |
| save_clips | - |
| sample_frames | 1.620 |
| caption_frames | 45.483 |
| sample_fps | 2.539 |
| detect_object_yolo | 9.136 |
| audio_scan | 14.851 |
| asr_timings | 8.257 |
| ast_timings | 31.960 |
| describe_scenes | 18.417 |
| summarize_scenes | 13.065 |
| synthesize_synopsis | 18.095 |
| make_embedding | 3.863 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 47.110 |
| branch_yolo_total | 11.681 |
| branch_audio_total | 55.077 |

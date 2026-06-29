# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-21 20:53:39 UTC | 00DH3yn5C30_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 0.060 | - | - | - | - | - | - |
| 2026-06-22 12:07:26 UTC | 00DH3yn5C30_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 132.467 | 0.772 | 44.587 | 14.618 | 8.370 | 18.971 | 2.888 |

## 2026-06-21 20:53:39 UTC | 00DH3yn5C30_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/00DH3yn5C30_210.0_360.0.mp4`
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

## 2026-06-22 12:07:26 UTC | 00DH3yn5C30_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/00DH3yn5C30_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `132.467` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.772 |
| save_clips | - |
| sample_frames | 0.730 |
| caption_frames | 29.631 |
| sample_fps | 2.101 |
| detect_object_yolo | 8.350 |
| audio_scan | 12.901 |
| asr_timings | 9.985 |
| ast_timings | 21.694 |
| describe_scenes | 14.618 |
| summarize_scenes | 8.370 |
| synthesize_synopsis | 18.971 |
| make_embedding | 2.888 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 30.367 |
| branch_yolo_total | 10.456 |
| branch_audio_total | 44.587 |

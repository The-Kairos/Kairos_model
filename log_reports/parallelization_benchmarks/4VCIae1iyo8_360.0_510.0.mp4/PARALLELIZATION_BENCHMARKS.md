# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 17:11:33 UTC | 4VCIae1iyo8_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 194.506 | 0.809 | 70.086 | 18.886 | 20.922 | 19.922 | 4.184 |
| 2026-06-24 11:04:53 UTC | 4VCIae1iyo8_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 182.281 | 0.801 | 65.465 | 16.482 | 15.960 | 16.910 | 4.182 |

## 2026-06-23 17:11:33 UTC | 4VCIae1iyo8_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/4VCIae1iyo8_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `194.506` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.809 |
| save_clips | - |
| sample_frames | 1.232 |
| caption_frames | 44.897 |
| sample_fps | 2.334 |
| detect_object_yolo | 9.842 |
| audio_scan | 11.658 |
| asr_timings | 23.093 |
| ast_timings | 35.327 |
| describe_scenes | 18.886 |
| summarize_scenes | 20.922 |
| synthesize_synopsis | 19.922 |
| make_embedding | 4.184 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 46.135 |
| branch_yolo_total | 12.181 |
| branch_audio_total | 70.086 |

## 2026-06-24 11:04:53 UTC | 4VCIae1iyo8_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/4VCIae1iyo8_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `182.281` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.801 |
| save_clips | - |
| sample_frames | 1.265 |
| caption_frames | 47.106 |
| sample_fps | 2.387 |
| detect_object_yolo | 10.262 |
| audio_scan | 11.833 |
| asr_timings | 17.967 |
| ast_timings | 35.656 |
| describe_scenes | 16.482 |
| summarize_scenes | 15.960 |
| synthesize_synopsis | 16.910 |
| make_embedding | 4.182 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 48.378 |
| branch_yolo_total | 12.655 |
| branch_audio_total | 65.465 |

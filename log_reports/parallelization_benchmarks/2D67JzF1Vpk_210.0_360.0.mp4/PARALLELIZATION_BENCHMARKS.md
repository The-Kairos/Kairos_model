# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-21 09:31:38 UTC | 2D67JzF1Vpk_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 181.652 | 1.860 | 72.512 | 10.063 | 14.406 | 10.818 | 4.251 |
| 2026-06-21 20:58:58 UTC | 2D67JzF1Vpk_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 179.159 | 1.924 | 70.777 | 12.786 | 10.405 | 8.306 | 4.284 |
| 2026-06-22 13:48:40 UTC | 2D67JzF1Vpk_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 223.756 | 1.891 | 68.538 | 31.249 | 21.866 | 25.165 | 6.176 |

## 2026-06-21 09:31:38 UTC | 2D67JzF1Vpk_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/2D67JzF1Vpk_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `181.652` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.860 |
| save_clips | - |
| sample_frames | 4.547 |
| caption_frames | 45.119 |
| sample_fps | 7.505 |
| detect_object_yolo | 9.260 |
| audio_scan | 15.780 |
| asr_timings | 21.608 |
| ast_timings | 35.116 |
| describe_scenes | 10.063 |
| summarize_scenes | 14.406 |
| synthesize_synopsis | 10.818 |
| make_embedding | 4.251 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 49.671 |
| branch_yolo_total | 16.771 |
| branch_audio_total | 72.512 |

## 2026-06-21 20:58:58 UTC | 2D67JzF1Vpk_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/2D67JzF1Vpk_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `179.159` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.924 |
| save_clips | - |
| sample_frames | 4.734 |
| caption_frames | 46.389 |
| sample_fps | 7.743 |
| detect_object_yolo | 10.349 |
| audio_scan | 16.321 |
| asr_timings | 18.589 |
| ast_timings | 35.858 |
| describe_scenes | 12.786 |
| summarize_scenes | 10.405 |
| synthesize_synopsis | 8.306 |
| make_embedding | 4.284 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 51.128 |
| branch_yolo_total | 18.098 |
| branch_audio_total | 70.777 |

## 2026-06-22 13:48:40 UTC | 2D67JzF1Vpk_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/2D67JzF1Vpk_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `223.756` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.891 |
| save_clips | - |
| sample_frames | 4.682 |
| caption_frames | 45.369 |
| sample_fps | 7.709 |
| detect_object_yolo | 9.712 |
| audio_scan | 16.129 |
| asr_timings | 16.554 |
| ast_timings | 35.846 |
| describe_scenes | 31.249 |
| summarize_scenes | 21.866 |
| synthesize_synopsis | 25.165 |
| make_embedding | 6.176 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 50.057 |
| branch_yolo_total | 17.426 |
| branch_audio_total | 68.538 |

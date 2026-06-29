# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 17:14:26 UTC | 4VCIae1iyo8_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 171.385 | 0.760 | 56.749 | 13.898 | 13.599 | 18.788 | 4.232 |
| 2026-06-24 11:07:45 UTC | 4VCIae1iyo8_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 169.875 | 0.783 | 55.667 | 15.927 | 18.947 | 11.103 | 4.200 |

## 2026-06-23 17:14:26 UTC | 4VCIae1iyo8_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/4VCIae1iyo8_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `171.385` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.760 |
| save_clips | - |
| sample_frames | 1.200 |
| caption_frames | 48.083 |
| sample_fps | 2.333 |
| detect_object_yolo | 10.298 |
| audio_scan | 10.707 |
| asr_timings | 10.009 |
| ast_timings | 36.025 |
| describe_scenes | 13.898 |
| summarize_scenes | 13.599 |
| synthesize_synopsis | 18.788 |
| make_embedding | 4.232 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 49.290 |
| branch_yolo_total | 12.637 |
| branch_audio_total | 56.749 |

## 2026-06-24 11:07:45 UTC | 4VCIae1iyo8_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/4VCIae1iyo8_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `169.875` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.783 |
| save_clips | - |
| sample_frames | 1.203 |
| caption_frames | 48.139 |
| sample_fps | 2.356 |
| detect_object_yolo | 10.105 |
| audio_scan | 10.707 |
| asr_timings | 9.149 |
| ast_timings | 35.802 |
| describe_scenes | 15.927 |
| summarize_scenes | 18.947 |
| synthesize_synopsis | 11.103 |
| make_embedding | 4.200 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 49.348 |
| branch_yolo_total | 12.467 |
| branch_audio_total | 55.667 |

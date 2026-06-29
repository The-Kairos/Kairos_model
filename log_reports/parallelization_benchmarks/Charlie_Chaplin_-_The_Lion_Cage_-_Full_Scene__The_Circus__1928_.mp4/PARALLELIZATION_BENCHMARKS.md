# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-05-13 06:53:31 UTC | Charlie_Chaplin_-_The_Lion_Cage_-_Full_Scene__The_Circus__1928_.mp4 | parallel | gemini | gemini-embedding-001 | 154.790 | 1.963 | 97.787 | 18.894 | 16.176 | 14.379 | 0.924 |

## 2026-05-13 06:53:31 UTC | Charlie_Chaplin_-_The_Lion_Cage_-_Full_Scene__The_Circus__1928_.mp4 | parallel

- Video path: `/home/Kairos_model/.tmp/kairos/jobs/7b17e830-f350-4fb1-9394-5d3d796a4a93/Charlie_Chaplin_-_The_Lion_Cage_-_Full_Scene__The_Circus__1928_.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `154.790` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.963 |
| save_clips | - |
| sample_frames | 0.224 |
| caption_frames | 6.373 |
| sample_fps | 9.238 |
| detect_object_yolo | 7.003 |
| audio_scan | 13.707 |
| asr_timings | 84.070 |
| ast_timings | 4.202 |
| describe_scenes | 18.894 |
| summarize_scenes | 16.176 |
| synthesize_synopsis | 14.379 |
| make_embedding | 0.924 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 6.605 |
| branch_yolo_total | 16.249 |
| branch_audio_total | 97.787 |

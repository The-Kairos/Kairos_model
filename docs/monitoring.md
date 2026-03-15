# 20: Monitoring and Rerunning Video Processing

This guide provides the exact commands to track your progress and instructions on how to restart or resume if the VM stops.

---

## 1. How to Monitor Progress (Any Video)

Replace `<VIDEO_DIR>` with your folder (e.g., `Titanic.1997.mkv` or `CCTV Dogs.mp4`).

### Stage-by-Stage Progress
| Pipeline Step | Tracking Command | Completion Target |
| :--- | :--- | :--- |
| **1. Scene Detection** | `ls -1 _processed/<VIDEO_DIR>/.clips | wc -l` | Total Scenes |
| **2. Frame Sampling** | `ls -1 _processed/<VIDEO_DIR>/.frames | wc -l` | Total Scenes |
| **3. YOLO Sampling** | `ls -1 _processed/<VIDEO_DIR>/.fps | wc -l` | Total Scenes |
| **4. LLM Description** | `grep -c "llm_scene_description" _processed/<VIDEO_DIR>/checkpoint.json` | Total Scenes |

### "Am I still running?"
```bash
ps aux | grep "python main.py" | grep -v grep
```

---

## 2. How to Rerun / Resume

### **Scenario A: The VM Crashed or Pre-empted**
**Action**: Just run the normal command again WITHOUT any redo flags.
```bash
python main.py process --video ".Titanic.1997.mkv"
```
**Why?** The script picks up where it left off automatically.

### **Scenario B: Forcibly restart from a step**
Use `--redo <STEP>` to restart from a specific phase.

| Restart From... | Flag |
| :--- | :--- |
| **Beginning** | `--redo scenes` |
| **Audio** | `--redo audio_speech` |
| **Objects (YOLO)** | `--redo yolo` |
| **Descriptions (LLM)** | `--redo llm` |
| **Final Narrative** | `--redo narrative` |

**Example**:
```bash
python main.py process --video ".Titanic.1997.mkv" --redo audio_speech
```

---

## 3. Important Tips
- **Automatic Resume**: Use this for 99% of crashes.
- **Redo Only**: Use `--redo-only narrative` if you just want to re-run the final summary.

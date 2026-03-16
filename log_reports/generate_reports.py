import os
import json
import subprocess

def format_time(seconds):
    seconds = float(seconds)
    if seconds < 60:
        return f"{int(seconds)} sec"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        if secs > 0:
            return f"{mins} min {secs} sec"
        return f"{mins} min"
    else:
        hours = int(seconds // 3600)
        rem = seconds % 3600
        mins = int(rem // 60)
        if mins > 0:
            return f"{hours} hour {mins} min"
        return f"{hours} hour"

def get_git_checkpoint(folder_name):
    try:
        git_path = f'_processed/{folder_name}/checkpoint.json'
        result = subprocess.run(['git', 'show', f'32859004818cbd7e8d28dada687eff0d4cf3ab5e:{git_path}'], 
                              capture_output=True, text=True, encoding='utf-8')
        if result.returncode == 0:
            return json.loads(result.stdout)
    except Exception as e:
        pass
    return None

def main():
    base_dir = r"c:\Users\tehre\OneDrive\Desktop\COMP4201-3 Capstone Project\Kairos Model\Kairos\_processed"
    folders = [f.name for f in os.scandir(base_dir) if f.is_dir()]
    
    results = []
    justifications = []
    malala_anomaly = {}
    
    for folder in folders:
        if folder == "Malala Yousafzai FULL Nobel Peace Prize Lecture 2014.mp4":
            continue
            
        local_cp_path = os.path.join(base_dir, folder, 'checkpoint.json')
        if not os.path.exists(local_cp_path):
            continue
            
        with open(local_cp_path, 'r', encoding='utf-8') as f:
            local_cp = json.load(f)
            
        git_cp = get_git_checkpoint(folder)
        if not git_cp:
            continue
            
        # Extract timings
        old_asr = git_cp.get('steps', {}).get('asr_timings', {}).get('wall_time_sec')
        new_asr = local_cp.get('steps', {}).get('asr_timings', {}).get('wall_time_sec')
        
        old_ast = git_cp.get('steps', {}).get('ast_timings', {}).get('wall_time_sec')
        new_ast = local_cp.get('steps', {}).get('ast_timings', {}).get('wall_time_sec')
        
        video_length = local_cp.get('video_length', 'Unknown')
        
        # Check if identical for Malala
        if "Malala Yousafzai" in folder and "FULL" in folder:
            malala_anomaly = {
                "git_start": git_cp.get("start_process"),
                "git_end": git_cp.get("end_process"),
                "local_start": local_cp.get("start_process"),
                "local_end": local_cp.get("end_process"),
                "git_run_desc": git_cp.get("run_description"),
                "local_run_desc": local_cp.get("run_description")
            }
            
        if old_asr and new_asr and old_ast and new_ast:
            asr_speedup = old_asr / new_asr
            ast_speedup = old_ast / new_ast
            
            results.append({
                'video': folder,
                'video_length': video_length,
                'old_asr': old_asr,
                'new_asr': new_asr,
                'asr_speedup': asr_speedup,
                'old_ast': old_ast,
                'new_ast': new_ast,
                'ast_speedup': ast_speedup
            })
            
            # Justifications
            justification = f"**Video:** {folder}\n"
            justification += f"- **Legacy processing timings** extracted from git commit `32859004818cbd7e8d28dada687eff0d4cf3ab5e`. [View File in Repo](https://github.com/The-Kairos/Kairos_model/blob/32859004818cbd7e8d28dada687eff0d4cf3ab5e/_processed/{folder.replace(' ', '%20')}/checkpoint.json)\n"
            justification += f"  - ASR Wall Time: {old_asr} seconds\n"
            justification += f"  - AST Wall Time: {old_ast} seconds\n"
            justification += f"- **New parallel processing timings** extracted from local file `_processed\\{folder}\\checkpoint.json`\n"
            justification += f"  - ASR Wall Time: {new_asr} seconds\n"
            justification += f"  - AST Wall Time: {new_ast} seconds\n"
            justification += "\n"
            justifications.append(justification)
            
    # Generate Table Report
    report_md = "# Comprehensive Audio Performance Report\n\n"
    report_md += "This report evaluates the performance improvements of the parallel audio processing pipeline (ASR and AST) compared to its sequential legacy counterpart.\n\n"
    report_md += "## Performance Metrics\n"
    report_md += "Using the standard speedup formula $S = \\frac{T_s}{T_p}$, where $T_s$ is the sequential execution time (legacy) and $T_p$ is the parallel execution time (new).\n\n"
    report_md += "**Note**: Timings below are estimated wall times extracted from checkpoints. Times have been rounded and formatted for human readability.\n\n"
    report_md += "| Video | Video Length | Legacy ASR Time | New ASR Time | ASR Speedup | Legacy AST Time | New AST Time | AST Speedup |\n"
    report_md += "|-------|--------------|-----------------|--------------|-------------|-----------------|--------------|-------------|\n"
    
    for r in results:
        report_md += f"| {r['video']} "
        report_md += f"| {r['video_length']} "
        report_md += f"| {format_time(r['old_asr'])} "
        report_md += f"| {format_time(r['new_asr'])} "
        report_md += f"| {r['asr_speedup']:.2f}x "
        report_md += f"| {format_time(r['old_ast'])} "
        report_md += f"| {format_time(r['new_ast'])} "
        report_md += f"| {r['ast_speedup']:.2f}x |\n"

    # Write main report
    os.makedirs(r"c:\Users\tehre\OneDrive\Desktop\COMP4201-3 Capstone Project\Kairos Model\Kairos\log_reports", exist_ok=True)
    with open(r"c:\Users\tehre\OneDrive\Desktop\COMP4201-3 Capstone Project\Kairos Model\Kairos\log_reports\audio_performance_report.md", 'w', encoding='utf-8') as f:
        f.write(report_md)
        
    # Write justifications
    just_md = "# Timing Justifications for Audio Performance Report\n\n"
    just_md += "The execution timings are estimated wall times precisely extracted from the `checkpoint.json` files generated after pipeline execution. They represent the estimated `wall_time_sec` recorded for the `asr_timings` and `ast_timings` steps.\n\n"
    just_md += "## Sources\n\n"
    for j in justifications:
        just_md += j
        
    with open(r"c:\Users\tehre\OneDrive\Desktop\COMP4201-3 Capstone Project\Kairos Model\Kairos\log_reports\audio_performance_timing_justification.md", 'w', encoding='utf-8') as f:
        f.write(just_md)
        
    with open('malala_debug.json', 'w') as f:
        json.dump(malala_anomaly, f)

    print("Reports successfully generated!")

if __name__ == '__main__':
    main()

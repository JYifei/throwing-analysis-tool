import os
import json
import pandas as pd
from pathlib import Path

# ==========================================
# 🎯 Adjust your score thresholds here
# ==========================================
K_VAL = 0.25  # Full score line (error <= K_VAL gets 100)
L_VAL = 4.00  # Zero score line (error >= L_VAL gets 0)
# ==========================================

def update_scores(runs_root: str, k: float, l: float):
    runs_path = Path(runs_root).resolve()
    if not runs_path.exists():
        print(f"[ERROR] Directory not found: {runs_path}")
        return 0

    count = 0
    for run_dir in runs_path.iterdir():
        if not run_dir.is_dir(): continue
        
        throws_dir = run_dir / "throws"
        if not throws_dir.exists(): continue
            
        for throw_dir in throws_dir.iterdir():
            if not throw_dir.is_dir() or not throw_dir.name.startswith("throw_"): continue
            
            score_path = throw_dir / "score.json"
            if not score_path.exists(): continue
                
            try:
                with open(score_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # Extract the original mean error
                mean_err = data.get("mean_normalized_error")
                if mean_err is None: continue
                
                # Recalculate the matching percentage (score)
                if mean_err <= k:
                    matching = 100.0
                elif mean_err >= l:
                    matching = 0.0
                else:
                    matching = 100.0 * (l - mean_err) / (l - k)
                
                # Update the JSON dictionary
                data["matching_percent"] = round(matching, 1)
                if "mapping" not in data:
                    data["mapping"] = {}
                data["mapping"]["k_full_score"] = k
                data["mapping"]["l_zero_score"] = l
                data["mapping"]["rule"] = f"e<={k} ->100; {k}<e<{l} -> linear; e>={l} ->0"
                
                # Overwrite and save
                with open(score_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                
                count += 1
            except Exception as e:
                print(f"[WARN] Failed to process {score_path}: {e}")

    return count

def generate_summary(runs_root: str, output_csv: str):
    runs_path = Path(runs_root).resolve()
    summary_data = []

    for run_dir in runs_path.iterdir():
        if not run_dir.is_dir(): continue
        throws_dir = run_dir / "throws"
        if not throws_dir.exists(): continue
            
        for throw_dir in throws_dir.iterdir():
            if not throw_dir.is_dir() or not throw_dir.name.startswith("throw_"): continue
            
            throw_id = throw_dir.name
            row = {"Run": run_dir.name, "Throw": throw_id}

            # 1. Read score
            score_path = throw_dir / "score.json"
            if score_path.exists():
                try:
                    with open(score_path, "r", encoding="utf-8") as f:
                        s_data = json.load(f)
                        row["Score"] = s_data.get("matching_percent", "")
                        row["Mean_Error"] = s_data.get("mean_normalized_error", "")
                except: pass

            # 2. Read criteria
            criteria_data = None
            for jf in ["criteria.json", "eval.json"]:
                jf_path = throw_dir / jf
                if jf_path.exists():
                    try:
                        with open(jf_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            if "items" in data: criteria_data = data["items"]
                            elif "criteria" in data: criteria_data = data["criteria"]
                            elif isinstance(data, list) and len(data) > 0 and "key" in data[0]: criteria_data = data
                            if criteria_data: break
                    except: pass
            
            if not criteria_data:
                for c in ["C1", "C2", "C3", "C4"]:
                    row[f"{c}_Pass"] = "-"
                    row[f"{c}_Reason"] = "No Data"
                summary_data.append(row)
                continue

            for item in criteria_data:
                key = item.get("key")
                if not key: continue
                student_result = item.get("student", {})
                passed = student_result.get("passed", False)
                notes = student_result.get("notes", [])
                
                row[f"{key.upper()}_Pass"] = "Y" if passed else "N"
                row[f"{key.upper()}_Reason"] = "" if passed else "; ".join(notes)
            
            summary_data.append(row)

    if not summary_data: return
        
    df = pd.DataFrame(summary_data)
    cols = ["Run", "Throw", "Score", "Mean_Error"]
    for c in ["C1", "C2", "C3", "C4"]:
        if f"{c}_Pass" in df.columns: cols.append(f"{c}_Pass")
        if f"{c}_Reason" in df.columns: cols.append(f"{c}_Reason")
    
    df = df[[c for c in cols if c in df.columns]].sort_values(by=["Run", "Throw"])
    
    out_path = Path(output_csv).resolve()
    try:
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"[SUCCESS] Generated latest summary report: {out_path}")
    except PermissionError:
        print(f"\n[WRITE FAILED] The file {out_path} is currently in use. Please close it in Excel and try again!")

if __name__ == "__main__":
    runs_dir = "runs"
    csv_name = "batch_summary_report.csv"
    
    print(f"Start updating scores with new thresholds (k={K_VAL}, l={L_VAL})...")
    updated_count = update_scores(runs_dir, k=K_VAL, l=L_VAL)
    print(f"Successfully updated json scores for {updated_count} videos in total.")
    
    if updated_count > 0:
        print("Regenerating CSV report...")
        generate_summary(runs_dir, csv_name)
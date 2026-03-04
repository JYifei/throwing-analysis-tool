# generate_summary.py
import os
import json
import pandas as pd
from pathlib import Path
import argparse

def summarize_all_runs(runs_root: str = "runs", output_csv: str = "batch_summary_report.csv") -> None:
    runs_path = Path(runs_root).resolve()
    if not runs_path.exists():
        print(f"[错误] 找不到运行记录文件夹: {runs_path}")
        return

    summary_data = []

    # 遍历 runs 文件夹下的所有子文件夹 (不论是 T1, 还是 时间戳)
    for run_dir in runs_path.iterdir():
        if not run_dir.is_dir():
            continue
        
        run_name = run_dir.name
        throws_dir = run_dir / "throws"
        
        if not throws_dir.exists():
            continue
            
        # 遍历每个 throw_xxx 文件夹
        for throw_dir in throws_dir.iterdir():
            if not throw_dir.is_dir() or not throw_dir.name.startswith("throw_"):
                continue
                
            throw_id = throw_dir.name
            
            # 初始化这一行的数据
            row = {"Run": run_name, "Throw": throw_id}

            # --- 1) 读取评分 (Score) ---
            score_path = throw_dir / "score.json"
            score_val = ""
            if score_path.exists():
                try:
                    with open(score_path, "r", encoding="utf-8") as f:
                        s_data = json.load(f)
                        val = s_data.get("matching_percent")
                        if val is not None:
                            score_val = str(val)
                except Exception:
                    pass
            row["Score"] = score_val

            # --- 2) 读取 Criteria ---
            json_files_to_check = ["criteria.json", "eval.json", "score.json"]
            criteria_data = None
            
            for jf in json_files_to_check:
                jf_path = throw_dir / jf
                if jf_path.exists():
                    try:
                        with open(jf_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            if "items" in data:
                                criteria_data = data["items"]
                                break
                            elif "criteria" in data:
                                criteria_data = data["criteria"]
                                break
                            elif isinstance(data, list) and len(data) > 0 and "key" in data[0]:
                                criteria_data = data
                                break
                    except Exception:
                        pass
            
            # 如果没找到 criteria 数据，填空
            if not criteria_data:
                for c in ["C1", "C2", "C3", "C4"]:
                    row[f"{c}_Pass"] = "-"
                    row[f"{c}_Reason"] = "No Data"
                summary_data.append(row)
                continue

            # 解析 C1 ~ C4
            for item in criteria_data:
                key = item.get("key")
                if not key:
                    continue
                
                student_result = item.get("student", {})
                passed = student_result.get("passed", False)
                notes = student_result.get("notes", [])
                
                row[f"{key.upper()}_Pass"] = "Y" if passed else "N"
                row[f"{key.upper()}_Reason"] = "" if passed else "; ".join(notes)
            
            summary_data.append(row)

    if not summary_data:
        print("[INFO] 没有找到可用于总结的数据（请检查 runs 文件夹内是否有 criteria.json）。")
        return
        
    df = pd.DataFrame(summary_data)
    
    # 动态排版列顺序: Run, Throw, Score, C1..., C2...
    cols = ["Run", "Throw", "Score"]
    for c in ["C1", "C2", "C3", "C4"]:
        if f"{c}_Pass" in df.columns: cols.append(f"{c}_Pass")
        if f"{c}_Reason" in df.columns: cols.append(f"{c}_Reason")
    
    valid_cols = [c for c in cols if c in df.columns]
    df = df[valid_cols]
    
    # 按文件夹名字字典序排序
    df = df.sort_values(by=["Run", "Throw"])
    
    out_path = Path(output_csv).resolve()
    
    # [关键]：加上异常捕捉，防止被 Excel 占用时直接崩溃
    try:
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"\n[成功] 所有结果已重新汇总至: {out_path}")
        print(f"共统计了 {len(df)} 条投掷记录。")
    except PermissionError:
        print(f"\n[写入失败] 文件 {out_path} 被占用！")
        print("💡 请先在 Excel 中关闭该文件，然后再重新运行此脚本。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="单独生成总结 CSV")
    parser.add_argument("--runs_root", type=str, default="runs", help="Runs 文件夹的路径 (默认: runs)")
    parser.add_argument("--output", type=str, default="batch_summary_report.csv", help="输出的 CSV 文件名")
    args = parser.parse_args()

    summarize_all_runs(runs_root=args.runs_root, output_csv=args.output)
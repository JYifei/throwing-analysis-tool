import os
import json
import pandas as pd
from pathlib import Path

def summarize_runs(runs_folder="runs", output_csv="summary_report.csv"):
    runs_path = Path(runs_folder)
    if not runs_path.exists():
        print(f"[错误] 找不到文件夹: {runs_folder}")
        return

    summary_data = []

    # 遍历 runs 文件夹下的所有子文件夹 (例如 T1, T2... T32)
    for run_dir in runs_path.iterdir():
        if not run_dir.is_dir():
            continue
        
        run_name = run_dir.name
        throws_dir = run_dir / "throws"
        
        if not throws_dir.exists():
            continue
            
        # 遍历每个 throw 文件夹 (例如 throw_001)
        for throw_dir in throws_dir.iterdir():
            if not throw_dir.is_dir() or not throw_dir.name.startswith("throw_"):
                continue
                
            throw_id = throw_dir.name
            
            # 寻找包含 criteria 的 json 文件
            json_files_to_check = ["score.json", "eval.json", "criteria.json"]
            criteria_data = None
            
            for jf in json_files_to_check:
                jf_path = throw_dir / jf
                if jf_path.exists():
                    try:
                        with open(jf_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            
                            # === 修改点：增加对 "items" 键的识别 ===
                            if "items" in data:
                                criteria_data = data["items"]
                                break
                            elif "criteria" in data:
                                criteria_data = data["criteria"]
                                break
                            # 有的直接是一个列表
                            elif isinstance(data, list) and len(data) > 0 and "key" in data[0]:
                                criteria_data = data
                                break
                    except Exception:
                        pass
            
            if not criteria_data:
                # 没找到数据，记录为异常
                summary_data.append({
                    "Run": run_name,
                    "Throw": throw_id,
                    "C1_Pass": "-", "C1_Reason": "No Data",
                    "C2_Pass": "-", "C2_Reason": "No Data",
                    "C3_Pass": "-", "C3_Reason": "No Data",
                    "C4_Pass": "-", "C4_Reason": "No Data",
                })
                continue

            # 初始化这一行的数据
            row = {
                "Run": run_name,
                "Throw": throw_id
            }
            
            # 解析 4 个指标
            for item in criteria_data:
                key = item.get("key")  # 'c1', 'c2', 'c3', 'c4'
                if not key:
                    continue
                
                student_result = item.get("student", {})
                passed = student_result.get("passed", False)
                notes = student_result.get("notes", [])
                
                pass_str = "Y" if passed else "N"
                # 如果是 N，就把 notes 里的原因用分号拼接起来；如果是 Y，原因留空
                reason_str = "" if passed else "; ".join(notes)
                
                row[f"{key.upper()}_Pass"] = pass_str
                row[f"{key.upper()}_Reason"] = reason_str
            
            summary_data.append(row)

    if not summary_data:
        print("[提示] 没有找到任何有效的指标评估数据。")
        return
        
    # 转换为 DataFrame
    df = pd.DataFrame(summary_data)
    
    # 排列列的顺序，确保是 Run, Throw, C1, C1_Reason, C2, C2_Reason...
    cols = ["Run", "Throw"]
    for c in ["C1", "C2", "C3", "C4"]:
        if f"{c}_Pass" in df.columns: cols.append(f"{c}_Pass")
        if f"{c}_Reason" in df.columns: cols.append(f"{c}_Reason")
    
    # 过滤掉不存在的列并应用顺序
    valid_cols = [c for c in cols if c in df.columns]
    df = df[valid_cols]
    
    # 聪明的排序：确保 T2 排在 T10 前面（自然排序）
    df['Run_Num'] = df['Run'].str.extract(r'(\d+)').astype(float)
    df = df.sort_values(by=['Run_Num', 'Throw']).drop(columns=['Run_Num'])
    
    # 导出为 CSV (加上 utf-8-sig 防止 Excel 打开中文乱码)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"\n[成功] 汇总报告已生成: {output_csv}")
    print(f"共统计了 {len(df)} 次投掷记录。")

if __name__ == "__main__":
    summarize_runs()
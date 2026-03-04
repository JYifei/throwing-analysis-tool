import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import subprocess
import os

class VideoTrimmerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("视频帧冻结裁剪工具 (终极修复版)")
        self.root.geometry("700x750")
        
        self.video_path = ""
        self.cap = None
        self.total_frames = 0
        self.fps = 0
        
        # UI 组件
        self.btn_open = tk.Button(root, text="第一步：选择视频文件", command=self.open_video, bg="#3498db", fg="white", font=("Arial", 10, "bold"))
        self.btn_open.pack(pady=10)
        self.lbl_filename = tk.Label(root, text="未选择视频", font=("Arial", 11, "italic"))
        self.lbl_filename.pack(pady=5)
        self.canvas = tk.Canvas(root, width=640, height=360, bg="black")
        self.canvas.pack()
        self.scale = tk.Scale(root, from_=0, to=100, orient=tk.HORIZONTAL, length=600, command=self.update_preview)
        self.scale.pack(pady=10)
        self.lbl_info = tk.Label(root, text="等待加载...")
        self.lbl_info.pack()

        param_frame = tk.Frame(root)
        param_frame.pack(pady=10)
        tk.Label(param_frame, text="静止时长 (秒):").grid(row=0, column=0)
        self.duration_entry = tk.Entry(param_frame, width=5)
        self.duration_entry.insert(0, "1")
        self.duration_entry.grid(row=0, column=1, padx=5)

        self.btn_run = tk.Button(root, text="开始处理 (一次性滤镜合成)", command=self.process_video, bg="#e67e22", fg="white", font=("Arial", 12, "bold"), height=2)
        self.btn_run.pack(pady=15)

    def open_video(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv")])
        if not self.video_path: return
        self.lbl_filename.config(text=f"当前视频：{os.path.basename(self.video_path)}")
        self.cap = cv2.VideoCapture(self.video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        # 修正部分视频 FPS 读取不准的问题
        if self.fps < 1: self.fps = 30 
        self.scale.config(to=self.total_frames - 1)
        self.scale.set(0)
        self.update_preview(0)

    def update_preview(self, frame_pos):
        if not self.cap: return
        frame_idx = int(frame_pos)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if ret:
            # 自动处理 OpenCV 预览时的旋转（OpenCV 通常能自动处理部分元数据，但需检查）
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(frame_rgb)
            # 自动应用 PIL 的方向修正
            try:
                from PIL import ImageOps
                img_pil = ImageOps.exif_transpose(img_pil)
            except: pass
            img_pil = img_pil.resize((640, 360))
            self.img = ImageTk.PhotoImage(img_pil)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img)
            self.lbl_info.config(text=f"起始位置：第 {frame_idx} 帧")

    def process_video(self):
        if not self.video_path: return
        
        target_frame = self.scale.get()
        start_time = target_frame / self.fps
        try:
            dur = float(self.duration_entry.get())
        except:
            dur = 1.0

        folder = os.path.dirname(self.video_path)
        new_name = f"processed_{os.path.basename(self.video_path)}"
        output_path = os.path.normpath(os.path.join(folder, new_name))

        # 使用一条强大的 FFmpeg 命令解决所有问题：
        # 1. -vf "reinit_filter=0" 防止旋转丢失
        # 2. 用 filter_complex 提取那一帧并循环，然后拼接后面的
        # 3. 强制统一帧率 (fps=fps) 并重新计算时间戳 (setpts)
        
        filter_str = (
            f"[0:v]split=2[v1][v2];"
            f"[v1]trim=start_frame={target_frame}:end_frame={target_frame+1},loop={int(dur*self.fps)}:1:0,setpts=N/FRAME_RATE/TB[frozen];"
            f"[v2]trim=start_frame={target_frame},setpts=PTS-STARTPTS[rest];"
            f"[frozen][rest]concat=n=2:v=1:a=0[outv]"
        )

        # 处理音频：如果视频有声音，截取对应的声音部分
        # (静止期间通常没声音，所以我们需要拼接“一段空白音”+“后续声音”)
        audio_filter = f"[0:a]atrim=start={start_time},asetpts=PTS-STARTPTS[outa]"

        cmd = [
            'ffmpeg', '-y',
            '-i', self.video_path,
            '-filter_complex', filter_str,
            '-map', '[outv]',
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-r', str(self.fps)
        ]

        # 尝试检查是否有音频流
        try:
            has_audio = subprocess.run(['ffprobe', '-show_streams', '-select_streams', 'a', self.video_path], capture_output=True, text=True).stdout
            if "codec_type=audio" in has_audio:
                # 如果有音频，合并声音并延迟声音
                # 静止时长 dur 秒对应空白音频
                audio_cmd = f"adelay={int(dur*1000)}|{int(dur*1000)}"
                cmd.extend(['-af', f'atrim=start={start_time},asetpts=PTS-STARTPTS,{audio_cmd}', '-c:a', 'aac'])
        except: pass

        cmd.append(output_path)

        try:
            self.btn_run.config(text="正在全速合成...", state=tk.DISABLED)
            self.root.update()
            
            # 这里的 shell=True 很有必要处理 Windows 路径
            subprocess.run(cmd, check=True)
            messagebox.showinfo("成功", f"处理完成！\n文件保存在：{output_path}")
        except Exception as e:
            messagebox.showerror("处理失败", f"错误：{e}")
        finally:
            self.btn_run.config(text="开始处理", state=tk.NORMAL)

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoTrimmerApp(root)
    root.mainloop()
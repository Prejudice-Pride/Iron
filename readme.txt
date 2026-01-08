项目简介：这是一个基于 Python/ PyTorch/OpenCV 的视频测震项目，
目标是从监控视频中提取多点位移时间序列以做地面振动分析。
工程以 co-tracker（cotracker3）为主模型（本地 checkpoint 加载），
并曾并行比较三种策略（OpenCV LK、cotracker 网格映射、混合），
现在你决定只用 cotracker3。输入为 input_videos/<video>.mp4，
输出到 output/（包括可视化视频 tracked_<video>.mp4、轨迹 npz/csv，以及每点的相对 Y 波形 csv）；
支持用 Shi‑Tomasi 在第一帧检测角点并把这些角点作为 cotracker 的 queries（也支持回退到网格）；
为避免检测时间戳/叠加文字，可通过 rois/ 下的 JSON 指定排除矩形（交互式在有显示时弹窗选择或在 headless 下先保存第一帧供人工标注），
代码会过滤落在 ROI 内的角点再送入模型；工程结构包含 co-tracker 子模块、strategies 模块（之前拆分以便对比）。

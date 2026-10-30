# ML case studies with results you can check

Three complete case studies, each run end to end on my own NVIDIA H100. The code, every raw prediction and the training logs are in this repository, so each number in the tables below can be recomputed.

| Project | What it shows | Key result |
|---|---|---|
| [LoRA vs QLoRA vs full fine-tuning](lora-vs-qlora-vs-full-finetuning/) | A controlled fine-tuning experiment with paired statistics | Qwen2.5-1.5B on Banking77 (3,080 test messages): LoRA 92.1%, QLoRA 91.5%, full fine-tuning 85.1%, base model 36.3% |
| [LLM screening for systematic reviews](llm-screening-systematic-reviews/) | Zero-shot LLM title and abstract screening; the threshold is set on one review and tested unchanged on two others | On the two test reviews: 88 of 92 final studies kept, 67% less manual reading (19,425 records in total) |
| [Robot demonstration data QC](lerobot-dataset-qc/) | Automatic quality checks for LeRobot teleoperation datasets, no manual labels | 158 episodes checked: 7 recordings where the task is never performed, 7.3% idle frames marked for trimming |

Code is MIT-licensed. Datasets keep their own licenses: Banking77 (PolyAI) is CC BY 4.0; the LeRobot and SYNERGY datasets are used under the terms on their own pages.

Stack: Python, PyTorch, Hugging Face Transformers and PEFT, bitsandbytes, pandas, SciPy, scikit-learn, ffmpeg.

I take on projects like these through Upwork: https://www.upwork.com/freelancers/~017960ad126f72864d

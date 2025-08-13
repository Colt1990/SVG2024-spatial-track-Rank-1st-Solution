# SVG2024-spatial-track Rank 1st Solution

This repository contains our competition entry for a multimodal generation challenge(SVG2024-spatial-track), where the goal is to generate high-quality videos and synchronized audio from random seeds.
Our work is built on top of the mmdiffusion framework and includes several important improvements to enhance both visual and audio fidelity.


# Key Improvements

## 1. Super-Resolution Model Retraining
	•	We extracted frames from the provided training dataset and used them to retrain a video super-resolution (SR) model.
	•	This SR model is applied after the base multimodal generation step to enhance spatial resolution and detail.

## 2. Optimized DPM-Solver Sampling
	•	We experimented with dpm_solver.sample parameters to achieve better perceptual quality and temporal consistency in generated videos.
	•	Adjustments include:
	•	steps: tuned to 80 for a good balance between quality and generation speed.
	•	order: evaluated both 2nd and 3rd order solvers, settled on order=2 for stability.
	•	skip_type: switched from "time_uniform" to "logSNR" for improved convergence.
	•	method: "singlestep" sampling for faster inference.
 ## 3. Additional Exploration
	•	Experimented with Real-ESRGAN for further quality enhancement, but results were not as effective as the retrained SR model.
	•	Considered architectural improvements to multimodal_unet, including:
	•	Enhancements to CrossAttentionBlock.
	•	RoPE (Rotary Position Embedding) integration for improved temporal modeling.
	•	These were not implemented in the final submission due to time constraints, but are promising future directions.

<div align="center">

<h1>Towards Robust Rain Removal Against Adversarial Attacks: A Comprehensive Benchmark Analysis and Beyond</h1>
<!-- # [CVPR 2022] Towards Robust Rain Removal Against Adversarial Attacks: A Comprehensive Benchmark Analysis and Beyond -->
</div>

[arxiv](https://arxiv.org/abs/2203.16931)

**Towards Robust Rain Removal Against Adversarial Attacks: A Comprehensive Benchmark Analysis and Beyond**
<br>_Yi Yu, [Wenhan Yang](https://flyywh.github.io/), Yap-Peng Tan, Alex C. Kot_<br>
In CVPR'22

## Datasets
We offer the test set of Rain100H `./data/test`and RainCityscapes  `./rain_cityscapes/test`.

For the full dataset, flease refer to [Rain100H](https://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html) and [RainCityscapes](https://team.inria.fr/rits/computer-vision/weather-augment/).

## Requirement
* GPU with memory size >= 24 GB
* pytorch==1.1 or higher version
* lpips

## Evaluate the robustness of our model on Rain100H
	cd ./code
	# LMSE attack with perturbation bound {1,2,4,8}
	python robust.py --data_test RainHeavyTest  --ext img --pre_train ../experiment/MPRNet_R_SEADD_MB_robust_pgd_Rain100H_e4/model/model_latest.pt --model MPRNet_R_SEADD_MB --test_only --save_results --save_gt --save_attack --save MPRNet_R_SEADD_MB_robust_pgd_test_Rain100H_e4_1 --n_GPUs 1 --attack_iters 20 --robust_epsilon 1 --robust_alpha 0.25
	python robust.py --data_test RainHeavyTest  --ext img --pre_train ../experiment/MPRNet_R_SEADD_MB_robust_pgd_Rain100H_e4/model/model_latest.pt --model MPRNet_R_SEADD_MB --test_only --save_results --save_gt --save_attack --save MPRNet_R_SEADD_MB_robust_pgd_test_Rain100H_e4_2 --n_GPUs 1 --attack_iters 20 --robust_epsilon 2 --robust_alpha 0.5
	python robust.py --data_test RainHeavyTest  --ext img --pre_train ../experiment/MPRNet_R_SEADD_MB_robust_pgd_Rain100H_e4/model/model_latest.pt --model MPRNet_R_SEADD_MB --test_only --save_results --save_gt --save_attack --save MPRNet_R_SEADD_MB_robust_pgd_test_Rain100H_e4_4 --n_GPUs 1 --attack_iters 20 --robust_epsilon 4 --robust_alpha 1
	python robust.py --data_test RainHeavyTest  --ext img --pre_train ../experiment/MPRNet_R_SEADD_MB_robust_pgd_Rain100H_e4/model/model_latest.pt --model MPRNet_R_SEADD_MB --test_only --save_results --save_gt --save_attack --save MPRNet_R_SEADD_MB_robust_pgd_test_Rain100H_e4_8 --n_GPUs 1 --attack_iters 20 --robust_epsilon 8 --robust_alpha 2

	# LPIPS attack with perturbation bound {1,2,4,8}
	python robust.py --data_test RainHeavyTest  --ext img --pre_train ../experiment/MPRNet_R_SEADD_MB_robust_pgd_Rain100H_e4/model/model_latest.pt --model MPRNet_R_SEADD_MB --test_only --save_results --save_gt --save_attack --save MPRNet_R_SEADD_MB_robust_pgd_test_Rain100H_e4_1_lpips --n_GPUs 1 --attack_iters 20 --robust_epsilon 1 --robust_alpha 0.25 --attack_loss lpips
	python robust.py --data_test RainHeavyTest  --ext img --pre_train ../experiment/MPRNet_R_SEADD_MB_robust_pgd_Rain100H_e4/model/model_latest.pt --model MPRNet_R_SEADD_MB --test_only --save_results --save_gt --save_attack --save MPRNet_R_SEADD_MB_robust_pgd_test_Rain100H_e4_2_lpips --n_GPUs 1 --attack_iters 20 --robust_epsilon 2 --robust_alpha 0.5 --attack_loss lpips
	python robust.py --data_test RainHeavyTest  --ext img --pre_train ../experiment/MPRNet_R_SEADD_MB_robust_pgd_Rain100H_e4/model/model_latest.pt --model MPRNet_R_SEADD_MB --test_only --save_results --save_gt --save_attack --save MPRNet_R_SEADD_MB_robust_pgd_test_Rain100H_e4_4_lpips --n_GPUs 1 --attack_iters 20 --robust_epsilon 4 --robust_alpha 1 --attack_loss lpips
	python robust.py --data_test RainHeavyTest  --ext img --pre_train ../experiment/MPRNet_R_SEADD_MB_robust_pgd_Rain100H_e4/model/model_latest.pt --model MPRNet_R_SEADD_MB --test_only --save_results --save_gt --save_attack --save MPRNet_R_SEADD_MB_robust_pgd_test_Rain100H_e4_8_lpips --n_GPUs 1 --attack_iters 20 --robust_epsilon 8 --robust_alpha 2 --attack_loss lpips


## Evaluate the robustness of our model on RainCityscape
	cd ./code
	# LMSE attack with perturbation bound {1,2,4,8}
	python robust.py --data_test RainHeavyTest  --ext img --pre_train ../experiment/MPRNet_R_SEADD_MB_robust_pgd_RainCityscapes100mm_half_e4/model/model_latest.pt --model MPRNet_R_SEADD_MB --test_only --save_results --save_gt --save_attack --save MPRNet_R_SEADD_MB_robust_pgd_test_RainCityscapes100mm_half_e4_1 --n_GPUs 1 --attack_iters 20 --robust_epsilon 1 --robust_alpha 0.25 --branch_reduction 4 --dir_data ../rain_cityscapes --apath ../rain_cityscapes/test/small/ --dir_hr ../rain_cityscapes/test/small/norain --dir_lr ../rain_cityscapes/test/small/rain100mm
	python robust.py --data_test RainHeavyTest  --ext img --pre_train ../experiment/MPRNet_R_SEADD_MB_robust_pgd_RainCityscapes100mm_half_e4/model/model_latest.pt --model MPRNet_R_SEADD_MB --test_only --save_results --save_gt --save_attack --save MPRNet_R_SEADD_MB_robust_pgd_test_RainCityscapes100mm_half_e4_2 --n_GPUs 1 --attack_iters 20 --robust_epsilon 2 --robust_alpha 0.5 --branch_reduction 4 --dir_data ../rain_cityscapes --apath ../rain_cityscapes/test/small/ --dir_hr ../rain_cityscapes/test/small/norain --dir_lr ../rain_cityscapes/test/small/rain100mm
	python robust.py --data_test RainHeavyTest  --ext img --pre_train ../experiment/MPRNet_R_SEADD_MB_robust_pgd_RainCityscapes100mm_half_e4/model/model_latest.pt --model MPRNet_R_SEADD_MB --test_only --save_results --save_gt --save_attack --save MPRNet_R_SEADD_MB_robust_pgd_test_RainCityscapes100mm_half_e4_4 --n_GPUs 1 --attack_iters 20 --robust_epsilon 4 --robust_alpha 1 --branch_reduction 4 --dir_data ../rain_cityscapes --apath ../rain_cityscapes/test/small/ --dir_hr ../rain_cityscapes/test/small/norain --dir_lr ../rain_cityscapes/test/small/rain100mm
	python robust.py --data_test RainHeavyTest  --ext img --pre_train ../experiment/MPRNet_R_SEADD_MB_robust_pgd_RainCityscapes100mm_half_e4/model/model_latest.pt --model MPRNet_R_SEADD_MB --test_only --save_results --save_gt --save_attack --save MPRNet_R_SEADD_MB_robust_pgd_test_RainCityscapes100mm_half_e4_8 --n_GPUs 1 --attack_iters 20 --robust_epsilon 8 --robust_alpha 2 --branch_reduction 4 --dir_data ../rain_cityscapes --apath ../rain_cityscapes/test/small/ --dir_hr ../rain_cityscapes/test/small/norain --dir_lr ../rain_cityscapes/test/small/rain100mm

	# LPIPS attack with perturbation bound {1,2,4,8}
	python robust.py --data_test RainHeavyTest  --ext img --pre_train ../experiment/MPRNet_R_SEADD_MB_robust_pgd_RainCityscapes100mm_half_e4/model/model_latest.pt --model MPRNet_R_SEADD_MB --test_only --save_results --save_gt --save_attack --save MPRNet_R_SEADD_MB_robust_pgd_test_RainCityscapes100mm_half_e4_1_lpips --n_GPUs 1 --attack_iters 20 --robust_epsilon 1 --robust_alpha 0.25 --attack_loss lpips --branch_reduction 4 --dir_data ../rain_cityscapes --apath ../rain_cityscapes/test/small/ --dir_hr ../rain_cityscapes/test/small/norain --dir_lr ../rain_cityscapes/test/small/rain100mm
	python robust.py --data_test RainHeavyTest  --ext img --pre_train ../experiment/MPRNet_R_SEADD_MB_robust_pgd_RainCityscapes100mm_half_e4/model/model_latest.pt --model MPRNet_R_SEADD_MB --test_only --save_results --save_gt --save_attack --save MPRNet_R_SEADD_MB_robust_pgd_test_RainCityscapes100mm_half_e4_2_lpips --n_GPUs 1 --attack_iters 20 --robust_epsilon 2 --robust_alpha 0.5 --attack_loss lpips --branch_reduction 4 --dir_data ../rain_cityscapes --apath ../rain_cityscapes/test/small/ --dir_hr ../rain_cityscapes/test/small/norain --dir_lr ../rain_cityscapes/test/small/rain100mm
	python robust.py --data_test RainHeavyTest  --ext img --pre_train ../experiment/MPRNet_R_SEADD_MB_robust_pgd_RainCityscapes100mm_half_e4/model/model_latest.pt --model MPRNet_R_SEADD_MB --test_only --save_results --save_gt --save_attack --save MPRNet_R_SEADD_MB_robust_pgd_test_RainCityscapes100mm_half_e4_4_lpips --n_GPUs 1 --attack_iters 20 --robust_epsilon 4 --robust_alpha 1 --attack_loss lpips --branch_reduction 4 --dir_data ../rain_cityscapes --apath ../rain_cityscapes/test/small/ --dir_hr ../rain_cityscapes/test/small/norain --dir_lr ../rain_cityscapes/test/small/rain100mm
	python robust.py --data_test RainHeavyTest  --ext img --pre_train ../experiment/MPRNet_R_SEADD_MB_robust_pgd_RainCityscapes100mm_half_e4/model/model_latest.pt --model MPRNet_R_SEADD_MB --test_only --save_results --save_gt --save_attack --save MPRNet_R_SEADD_MB_robust_pgd_test_RainCityscapes100mm_half_e4_8_lpips --n_GPUs 1 --attack_iters 20 --robust_epsilon 8 --robust_alpha 2 --attack_loss lpips --branch_reduction 4 --dir_data ../rain_cityscapes --apath ../rain_cityscapes/test/small/ --dir_hr ../rain_cityscapes/test/small/norain --dir_lr ../rain_cityscapes/test/small/rain100mm

## Evaluate the result
The generated results are in the folder: `./experiments`, and you can evaluate the results by PSNR or SSIM. Images with suffix 'SR' are the clean outputs, images with suffix 'SR_attack' are the attacked outputs, images with suffix 'LR' are the clean inputs, images with suffix 'LR_attack' are the perturbed inputs, and images with suffix 'HR' are the groundtruth. For downstream tasks, please refer to the code of [SSeg](https://github.com/YeLyuUT/SSeg) and [Pedestron](https://github.com/hasanirtiza/Pedestron).

## Citation
If you find our work useful for your research, please consider citing this paper:
	@article{yu2022towards,
	  title={Towards Robust Rain Removal Against Adversarial Attacks: A Comprehensive Benchmark Analysis and Beyond},
	  author={Yu, Yi and Yang, Wenhan and Tan, Yap-Peng and Kot, Alex C},
	  journal={arXiv preprint arXiv:2203.16931},
	  year={2022}
	}
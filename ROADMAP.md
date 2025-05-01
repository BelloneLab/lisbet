# Roadmap

## 🔜 Q2 2025
- [ ] Add support for SLEAP ([#8, in progress](https://github.com/BelloneLab/lisbet/issues/8))
- [x] Add support for saving and loading HMMs ([#14](https://github.com/BelloneLab/lisbet/issues/14))
- [x] Speed up HMM fitting by training on a random subset of data ([#22](https://github.com/BelloneLab/lisbet/issues/22))
- [x] Add native support for parallel HMM scanning in CLI ([#23](https://github.com/BelloneLab/lisbet/issues/23))
- [x] Test training LISBET classifiers using HMM prototypes ([#24](https://github.com/BelloneLab/lisbet/issues/24))
- [ ] Testing “multi-dyadic” strategy for inference on groups (n > 2) ([#27](https://github.com/BelloneLab/lisbet/issues/27))

## 📅 Q3 2025
- [ ] Introduce "Switch Individuals" as a data augmentation strategy
- [ ] Add native support for n > 2 individuals ([#28](https://github.com/BelloneLab/lisbet/issues/28))
- [ ] Conduct ablation study on pretraining tasks

## 🔮 Future Ideas
- [ ] Explore end-to-end models for discovery-driven behavior labeling ([#21](https://github.com/BelloneLab/lisbet/issues/21))
- [ ] Add support for raw video input
- [ ] Add support for segmentation mask input
- [ ] Enable segmentation mask generation from keypoints
- [ ] Compare keypoints vs. keypoint-derived segmentation mask performance
- [ ] Evaluate frame-based models (raw video and segmentation masks) for n > 2 individuals
- [ ] Refactor CLI ([#20](https://github.com/BelloneLab/lisbet/issues/20))
- [ ] Introduce GUI
- [ ] Explore the use of LLMs for behavior classification from embeddings
- [ ] Automate model benchmarking

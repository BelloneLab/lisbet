# Roadmap

## 🔜 Q2 2025
- [ ] Add support for SLEAP ([#8, in progress](https://github.com/BelloneLab/lisbet/issues/8))
- [ ] Add support for saving and loading HMMs ([#14, in progress](https://github.com/BelloneLab/lisbet/issues/14))
- [ ] Accelerate HMM fitting via random sampling
- [ ] Enable parallel HMM scanning using multiprocessing
- [ ] Add support for fine-tuning classification models using HMM prototypes
- [ ] Add support for n > 2 individuals, "multi-dyadic" strategy

## 📅 Q3 2025
- [ ] Introduce "Switch Individuals" as a data augmentation strategy
- [ ] Add native support for n > 2 individuals
- [ ] Conduct ablation study on pretraining tasks

## 🔮 Future Ideas
- [ ] Implement end-to-end modeling (i.e., remove HMM stage)
- [ ] Add support for raw video input
- [ ] Add support for segmentation mask input
- [ ] Enable segmentation mask generation from keypoints
- [ ] Compare keypoints vs. keypoint-derived segmentation mask performance
- [ ] Evaluate frame-based models (raw video and segmentation masks) for n > 2 individuals
- [ ] Refactor CLI
- [ ] Introduce GUI
- [ ] Explore the use of LLMs for behavior classification from embeddings
- [ ] Automate model benchmarking

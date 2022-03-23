# 1 Dataset

This code can run on dataset combined of SAMM and CASME II.

All the MERs are divided into 3 classes:

- surprise
  - surprise
- positive
  - happiness
- negative
  - CASME II(disgust, fear,  sadness, repression) 
  - SAMM(disgust, fear, sadness, repression, anger)

The dataset should be organized as following:

```
.data_3
├── negative
│   ├── 006_1_2
│   ├── 006_1_3
│   ├── EP01_01
	├── ...
├── positive
│   ├── 007_6_1
│   ├── 007_6_2
│   ├── EP01_01f
	├── ...
├── surprise
│   ├── 006_3_5
│   ├── 006_5_9
│   ├── EP01_13
	├── ...
```

# 2 Images

- All frames in the dataset are cropped and resized to 128 * 128. 
- If you want to use different size of frames, change the parameter of CNN in `MicroExpSTCNN_torch.py`.
- You may need to make a dir `saved_models_3` to save the models, or you can change the dir name in `MicroExpSTCNN_torch.py`.

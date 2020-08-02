# Steel_Defect_Detection
competetion named "Steel Defect Detection" [on kaggle](https://www.kaggle.com/c/severstal-steel-defect-detection/notebooks)

# Data Visualizatoin
if you want to check total number of training data and visualize some of them

```
python3 ./datavis.py
```

# Training
you can choose following models and loss functions
please type following keywords in ()

models : -m (default = unet)
- UNet (unet)
- ResNet-based Unet (resnetx: x is num layers of ResNet)
- ResUNet-a ()

```
-m <MODEL NAME>
```

loss functions : -l (default = BCE)
- BCE (BCE)
- Dice (Dice)
- DiceBCE (DiceBCE)
- IoU (IoU)
- Focal (Focal)
- Tanimoto (Tanimoto)

```
-l <FUNCTION NAME>
```
Then, you can train model like this

```
$ cd src
$ python3 ./main.py -m <MODEL NAME> -l <FUNCTION NAME>
```

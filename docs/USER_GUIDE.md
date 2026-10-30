# Your first model: a walkthrough

This guide takes you from a folder of labelled images to a model you can use, and explains each decision on the
way. Run `mtrain gui`, and keep **Simple** mode on (top right) for your first run.

## 1. Import the dataset

**Datasets → Import dataset.** Paste the folder path. The importer recognises three layouts.

**YOLO** (most labelling tools export it):
```
my_dataset/
  data.yaml          names: {0: cat, 1: dog}, train: images/train, val: images/val
  images/train/*.jpg   labels/train/*.txt
  images/val/*.jpg     labels/val/*.txt
```

**COCO:** json files such as `annotations/train.json` and `val.json`, plus the images.

**Classification:** one folder per class, optionally inside `train/` and `val/`.

**Optional `groups.csv`.** If several photos show the same scene, add a `groups.csv` file with the columns
`image,group`. Those photos will never be split between training and validation.

## 2. Read the health report

Each card has a colour:
- **green:** fine;
- **blue:** a note;
- **orange:** worth attention;
- **red:** will hurt training.

Open a card to see what it means and what to do. The most important checks:

**Object size at the training image size.** Images are shrunk before training. If your objects become a few pixels
wide, no model will find them. Pick the size the report recommends.

**Class balance.** A class with ten times fewer examples is learned much worse. You have three options:
- add data for that class;
- turn on *oversample rare classes* in the wizard;
- merge or drop the class.

**Leakage.** The same photo in both the training and the validation set makes the scores look better than they
are. Fix it by re-splitting in the wizard.

**Label problems.** Click the card's filter in the **Gallery** to see the broken labels, and fix them in your
labelling tool. Then press **Re-read files**.

**What the model sees** shows real training samples after augmentation. Check that your objects are still
recognisable.

## 3. Set up the run (wizard)

1. **Task.** Only the tasks your labels allow can be picked. Masks can also train a box detector. Boxes can also
   train a classifier, which then works on cropped objects.
2. **Base model.** Start with the recommendation, usually YOLO26 *nano* or *small*. The table shows whether it fits
   in your GPU memory and roughly how long an epoch takes. Keep **pretrained**.
3. **Image size.** Use the health report's recommendation.
4. **Augmentation:**
   - **standard** for everyday photos;
   - **light** for a fixed camera and lighting (a rig);
   - **strong** for small datasets.

   If colour is what separates your classes, set hue and saturation to 0.
5. **Training.** Typical values:
   - **epochs:** 100;
   - **patience:** 30-50 (the run stops by itself when it stops improving);
   - **batch:** -1 (automatic).
6. **Review.** Read the estimates and the warnings, give the run a name and a note about what you are trying, then
   press **Start now**.

**Tip:** first run a *quick test* on 10-20 % of the images (step 1). If that run learns something, start the full
run.

## 4. While it trains

**Losses** should go down.
- If the **validation loss goes up while the training loss keeps falling**, the model is overfitting. You get a
  warning, and early stopping keeps the best epoch anyway.

**Scores** (mAP, precision, recall) should go up and then flatten.

**Watch it learn** shows the predictions on the same images after each epoch.

**Pause** frees the GPU at the end of the epoch. **Resume** continues exactly where it stopped. Training also
continues if you close the browser.

**If the computer crashes or loses power,** the run shows as *interrupted*. Press **Resume**.

**Phone notifications:** set up ntfy in **Settings** to get a message when a run finishes or fails.

## 5. Judge the result

**Results** starts with a plain sentence, for example "of every 100 objects the model finds about 93…".

**Per-class table:** look for classes far below the others.

**Confusion matrix:** shows which classes get mixed up. Click a cell to see those images.

**Error gallery:** the images with the most mistakes. It usually shows what to fix:
- missing labels;
- look-alike classes;
- objects that are too small.

**Best confidence threshold:** use this value when you deploy the model.

## 6. Try it for real

In the **Playground**, drop photos the model has **never seen**, ideally from the place where it will be used.
This is the real test.

**Compare two models:** choose two runs to see them side by side.

**Active learning:** *Send these images to a new dataset* saves the predictions as labels. Correct them, import
the result, and train again on the hard cases.

## 7. Export

**Export** creates a file for your target device:
- **ONNX:** anywhere;
- **NCNN:** a Raspberry Pi;
- **TensorRT:** an NVIDIA PC.

Each export is re-scored, so you can see whether FP16 or INT8 lost accuracy.

The `package/` folder holds everything another program needs, including `model_card.json` with the class list.

## When results are poor

| Symptom | Likely cause | Try |
|---|---|---|
| Small objects missed | Image size too small | Larger image size; check *object size* in the health report |
| One class always confused with another | They look alike, or have too few examples | More examples of both; check the labels; merge them if the difference does not matter |
| Great validation score, poor on real photos | Validation too similar to training (leakage), or training data unlike reality | Re-split by group; add real photos; stronger augmentation |
| Validation loss rises early | Overfitting | More data, stronger augmentation, smaller model |
| Scores still rising at the last epoch | Training too short | **Train longer** |
| "GPU out of memory" | Batch, image size or model too large | Batch -1 (auto), a smaller model, or a smaller image size |

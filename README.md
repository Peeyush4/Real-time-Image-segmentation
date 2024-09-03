## Each folder/file information

files/: This is the place where each model information along with .pth file and proof that we used models like PPLite B50, T50, STDC and DDRNet and all models' plots.
models/: This contains models Fast-SCNN and BiSeNet
results/: This is the place where model results are dumped
videos/: This is the place where videos are placed

config_v2.0.json: Configuration used by Mapillary
bisenet_video.py: File required to run BiSeNet model on videos
fast_scnn_video.py: File required to run Fast-SCNN model on videos
MSML612 Presentation.pptx: Our presentation
MSML612 Report: Report containing all the information
Presentation_video.mp4: Our presentation video explaining the presentation
requirements.txt: Contains which libraries to download


## Instructions 
Main python codes to run are bisenet_video.py and fast_scnn.py

You can run the codes for BiSeNet by running the following in the terminal
python bisenet_video.py -f <file_name> -d <directory> -m <model_name>

You can run the codes for Fast-SCNN by running the following in the terminal
python fast_scnn_video.py -f <file_name> -d <directory> -m <model_name>

Where the <file_name>, <directory> and <model_name> are as follows:
<file_name> is the file name from the videos/ folder,

<directory> is the name of the folder from the folder "files/" as this indicates the model to use. The following folders only
- bisenet_crossEntropyLoss_Weights
- bisenet_focalLoss
- bisenet_poly_lr_research_paper
- scnn_crossEntropyLoss
- scnn_crossEntropyLoss_Weights
- scnn_focalLoss_3gamma
- scnn_sgd_optimizer_poly

<model_name> is the .pth file inside the those respective directories in the <directory> folder. You can choose which .pth file to choose in the <directory> folder.

BiSeNet codes run only when bisenet directories are given to bisenet_video.py and Fast-SCNN run only when fast_scnn_video.py

Now, the example code is 
python bisenet_video.py -f transcodedVideo.mp4 -m bisenet_crossEntropyLoss_Weights_1e-4lr_60epochs_after_1e-lr_30_epochs -d bisenet_crossEntropyLoss_Weights

When the code runs, the result will be in the results folder with the name <file_name>_<model_name>.mp4 and the output will be similar to the below lines 
Number of frames: 601
Reading time: 0.29398131370544434
Transform time: 21.074201822280884
Model time: 2.932260751724243
Writing time: 40.23506021499634
Total time: 64.53550410270691


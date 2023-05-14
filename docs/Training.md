# Training
1. Follow instructions from Installation docs
<br/><br/>
2. Put raw card images into ```images/raw``` directory for card detection, also put icdar_2013_images into ```images``` directory for text detection
<br/><br/>
3. Place your ground truth files into annotations directory 
(i used CVAT for labeling and exported them as .xml file, icdar dataset is using .txt format). 
Make sure that names are in conformance with what you see/set up in ```prepare_datasets``` notebook
<br/><br/>
3. Run ```notebooks/data_partition.ipynb```
<br/><br/>
4. Run ```notebooks/prepare_datasets.ipynb```
<br/><br/>
5. Run ```notebooks/train_models.ipynb```
<br/><br/>

# GraphCXRNet
I implement CXR image detection using GraphCXRNet, a variety of Graph Neural Networks based architecture such as GCN, and GraphConv. The code has been performed on 3-class (in fection of Bacteria vs Normal vs Virus) dataset consisting of approximately 10,000 Chest X-ray images.

# Dataset link
```
1. Chest-Xray dataset: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia , CXR, 3-class

```
# File Structure and Working procedure
```
1. First, all CXR images need resizing to 224x224x1 by runing a code: `1_resize.py`

2. A second step, I apply edge transformation with a code: `2_Edge_transformation_Prewitt.py`

3. Then construct graph-datasets using a code: `3_Graph_construction_Prewitt.py`

4. Finally, graph data produces five kinds of dataset for graph classification:
  path name: .../GraphTrain/dataset/<dataset_name>/raw/<dataset_name>_<data_file>.txt. 
  <data_file> can be:
    
    1. A--> adjancency matrix 
    2. graph_indicator--> graph-ids of all node 
    3. graph_labels--> labels for all graph 
    4. node_attributes--> attribute(s) for all node 
    5. node_labels--> labels for all node

5. After all the graph datasets are created properly, run main.py. The graph datasets are loaded through dataloader.py and the model is called through models.py
```
# Checking results
```
1. You can check accuracy curves during training phases and final confusion matrix on test set as well by looking at the `results` folder.

2. You also reuse trained weights by refering the `weights` folder

3. To reimplement my whole experiment, you can download data from `releases` task.
```
# Reference
I have refered a paper: "GraphCovidNet: a graph neural network based model for detecting COVID‑19 from CT scans and X‑rays of chest" 
```
@article{saha2021GraphCovidNet,
  title={GraphCovidNet: a graph neural network based model for detecting COVID‑19 from CT scans and X‑rays of chest}
  author={Saha, Pritam and Mukherjee,  Debadyuti and Singh,Pawan and Ahmadian, ALi and Ferrara, Massimiliano and  Sarkar, Ram}
}
```

# @creator: Dương Tuấn Linh
# Please do not hesitate to contact me if you have any question!

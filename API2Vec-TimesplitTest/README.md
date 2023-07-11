# Quick Start - main.py

## Dataset

The dataset folder directory should contain two subfolders `black` and `white`
- The `black` folder store the API sequence information of malware
- The `white` folder store the API sequence information of goodware

Tips: API sequence information need to be processed manually as tabular file in `.xlsx` format, which must include the following header:
- apicall: name of a single API call
- pid: process ID of API call
- ppid: parent process ID of API call
- timestamp: timestamp of API call
- status: status information of API execution (`SUCCESS` or `FALIURE`)

## Data Process

Use the `./DataProcess/*` for data preprocessing: 
1. `./DataProcess/process_api_sequence`: process raw dataset for graph modeling and heuristic random walk
2. `./DataProcess/process_corpus`: graph modeling and heuristic random walk
3. `./DataProcess/api2vec`: generate embedding using doc2vec or word2vec
4. ``./DataProcess/process_dataloader`:  data encapsulation of subsequent malware detection models

## Malware Detection

```python
python main.py 2009_2018_2019_2019_cuck_pid_doc2vec_ml_knn_mean_status+count
```

- 2009_2018_2019_2019: Indicates that the year of the training  dataset is `[2009-2018]` the year of the test dataset is ` [2019, 2019]`
- cuck: dataset name
- pid: for API2Vec Model
  - option values: nopid, node2vec
  - nopid: for basic word2vec model
  - node2vec: for node2vec model
- doc2vec: for embedding method
  - option values: w2v
  - w2v: for word2vec method
- ml: model type, means machine learning
- knn: malware detection method
- mean: deprecated
- status+count: other parameters
  - status: consider the execution status of the API
  - count: for R2P mechanism, bind with `pid` only

# config.py

Various configuration information

## config.name2time

Record the compilation time information of the target sample: load from `./outputs/exp_name2time.pkl`

```python
{
  "0000cf924bcea1bf0cfbc79d30a9ef2d581e1834ad88df2d4e4460173a5e6dd3": {
    "year": 2021, 
    "month": 4
  },
  ...
}
```
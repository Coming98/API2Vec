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
python main.py normal_cuck_pid_doc2vec_ml_knn_mean_status+count
```

- normal: for malware detection
  - optional values: attack, target
  - attack: for adversarial attacks  
  - target: for concept drift of malware types
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
  - virus: target malware type, bind with `target` only; 
    - optional values: backdoor, worm, grayware, downloader

# config.py

Various configuration information

## config.infos

Data information processed offline: load from `./outputs/infos.pkl`

```python
{
  "0000cf924bcea1bf0cfbc79d30a9ef2d581e1834ad88df2d4e4460173a5e6dd3": {
    "label": 0, 
    "pid_count": 4
  },
  ...
}
```

## config.against_sample_names

Name list of test set in the adversarial attacks experiment: load from `./outputs/against_sample_names.pkl`

```python
[
  '8e11efca38fcbf07e22f152267d8e3e70e663f87ce3eda6bf379efbe6ee21530', '536c8b0021c95fdfb8c22a42557e59e1f17ecbee8a5c2ae9134416d65a41c16d',
  ...
]
```

## config.type_names

List of malware names for different malware types: load from `./outputs/type_names.pkl`

```python
{
  "virus": [
    '5f7a350ac754b300a8a770a5d0ae69eef6a2a227f36f948042394bb0a8603827', '883e6ac8fe852bf34bcc08a56c1237495cec1fa91b514d7d1f111c8ea4f78540',
    ...
  ],
  "backdoor": [
    'c836130702b103deeb5abb314984804d2bdf1fbc971ad61c1aefc11e3a697aeb', '266c34a2915ff96b761a228a7d0f09a414b7885f001b2bd64d8246c1a0f8ce05',
    ...
  ]
}
```
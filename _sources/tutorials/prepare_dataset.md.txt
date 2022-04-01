# Preparing data for training [TODO]

In `lidar_multiclass/datamodule/data.py` is the logic for data pre-processing, both offline and online, i.e. saving preprocessed data objects for fast trainig vs. pre-processing at inference time. 

The loading function is dataset dependant, and there are currently a logic for both SwissTopo data (withour infrared channel) and French IGN data (with infrared channel).

For help, run 

```
python lidar_multiclass/datamodules/data.py -h
```
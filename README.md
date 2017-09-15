# data2text

Code for [Challenges in Data-to-Document Generation](https://arxiv.org/abs/1707.08052) (Wiseman, Shieber, Rush; EMNLP 2017); much of this code is adapted from an earlier fork of [OpenNMT](https://github.com/OpenNMT/OpenNMT).

The boxscore-data associated with the above paper can be downloaded from the [boxscore-data repo](https://github.com/harvardnlp/boxscore-data), and this README will go over running experiments on the RotoWire portion of the data; running on the SBNation data (or other data) is quite similar.


## Preprocessing
Before training models, you must preprocess the data. Assuming the RotoWire json files reside at `~/Documents/code/boxscore-data/rotowire`, the following command will preprocess the data

```
th box_preprocess.lua -json_data_dir ~/Documents/code/boxscore-data/rotowire -save_data roto
```

and write files called roto-train.t7, roto.src.dict, and roto.tgt.dict to your local directory.

### Incorporating Pointer Information
For the "conditional copy" model, it is necessary to know where in the source table each target word may have been copied from.

This pointer information can be incorporated into the preprocessing by running:

```
th box_preprocess.lua -json_data_dir ~/Documents/code/boxscore-data/rotowire -save_data roto -ptr_fi "roto-ptrs.txt"
```

The file roto-ptrs.txt has been included in the repo, but you can also generate one with

```
python data_utils.py -mode ptrs -input_fi ~/Documents/code/boxscore-data/rotowire/train.json -output_fi "my-roto-ptrs.txt"
```

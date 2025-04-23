We use subsets of the Satlas pre-training sets because of memory constraints.

To download the subsets run: 

```wget https://ai2-public-datasets.s3.amazonaws.com/satlas/satlas-dataset-v1-naip-small.tar```
```wget https://ai2-public-datasets.s3.amazonaws.com/satlas/satlas-dataset-v1-sentinel2-small.tar```

and extract the datasets into their specific locations: 

```tar -xf satlas-dataset-v1-naip-small.tar -C satlas/naip/```
```tar -xf satlas-dataset-v1-sentinel2-small.tar -C satlas/sentinel2/```

Then, for NAIP and Sentinel-2 individually, move all images from their subfolders into one common folder:

```satlas/naip/naip_small/``` for NAIP, and
```satlas/sentinel2/sentinel2_small/``` for Sentinel-2.
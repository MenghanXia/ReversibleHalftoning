import json
from os.path import join
from glob import glob

colors = [join('HalftoneVOC2012', x) for x in glob(join('val/target_c', '*.png'))]
halfs = [x.replace('target_c', 'raw_ov') for x in colors]
valSet = {'inputs': colors, 'labels': halfs}

colors = [join('HalftoneVOC2012', x) for x in glob(join('train/target_c', '*.png'))]
halfs = [x.replace('target_c', 'raw_ov') for x in colors]
trainSet = {'inputs': colors, 'labels': halfs}

with open('../HalftoneVOC2012.json', 'w') as f:
    json.dump({'train': trainSet, 'val': valSet}, f)

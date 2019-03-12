import json
import numpy as np
import torch


#
# Some numpy types are not serializable to JSON out-of-the-box in Python3 -- need coersion. See
# http://stackoverflow.com/questions/27050108/convert-numpy-type-to-python/27050186#27050186
#

class JsonEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.nn.Parameter):
            return obj.data
        elif isinstance(obj, torch.Tensor):
            return obj.detach_().numpy()
        else:
            return super(JsonEncoder, self).default(obj)

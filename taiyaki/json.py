import json
import numpy as np
import torch


#
# Some numpy types are not serializable to JSON out-of-the-box in Python3 -- need coersion. See
# http://stackoverflow.com/questions/27050108/convert-numpy-type-to-python/27050186#27050186
#

class JsonEncoder(json.JSONEncoder):
    """ A custom JSON encoder with support for numpy arrays and torch tensors

    This encoder can be used instead of `json.JSONEncoder` from the standard
    library, with added support for numpy arrays and torch tensors, which
    will be converted to JSON arrays.

    This encoder can be used to serialise models built using the modules defined
    in `taiyaki.layers` in a guppy-compatible format.

    Examples:
        >>> import json
        >>> from taiyaki.json import JsonEncoder
        >>> from taiyaki.layers import FeedForward, Serial
        >>> model = Serial([FeedForward(1, 12), FeedForward(12, 32)])
        >>> json.dumps(model.json(), indent=2, cls=JsonEncoder)
        {
          "type": "serial",
          "sublayers": [
            {
              "type": "feed-forward",
              "activation": "linear",
              "size": 12,
              "insize": 1,
              "bias": true
            },
            {
              "type": "feed-forward",
              "activation": "linear",
              "size": 32,
              "insize": 12,
              "bias": true
            }
          ]
        }
    """

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

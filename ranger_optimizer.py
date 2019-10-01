# new custom state of the art optimizer.
# I had written my own implementation of another version of adam proposed in this paper - https://arxiv.org/abs/1711.05101,
# but the method linked in this file is absolutely state of the art (came out 1 month ago!)
# good thing is, there is already an implementation in tensorflow!

# nice explaination - https://medium.com/@lessw/new-deep-learning-optimizer-ranger-synergistic-combination-of-radam-lookahead-for-the-best-of-2dc83f79a48d

# implementation - pip install keras-rectified-adam==0.17.0
# codebase for above - https://github.com/tensorflow/addons/blob/master/tensorflow_addons/optimizers/rectified_adam.py

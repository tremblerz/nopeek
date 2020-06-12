from resnet import ResNetSplit, BasicBlock

def ResNetSplitClient():
    return ResNetSplit(BasicBlock, [3, 4, 6, 3])

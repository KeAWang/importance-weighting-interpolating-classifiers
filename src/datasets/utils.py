from collections import namedtuple


LabeledDatapoint = namedtuple("LabeledDatapoint", ("x", "y"))
GroupedLabeledDatapoint = namedtuple("GroupedLabeledDatapoint", ("x", "y", "g"))

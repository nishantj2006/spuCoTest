import torch
from spuco.datasets import SpuCoMNIST, SpuriousFeatureDifficulty
from spuco.models import model_factory
from torch.optim import SGD
from spuco.robust_train import ERM
from spuco.group_inference import Cluster, ClusterAlg
from spuco.robust_train import GroupBalanceBatchERM, ClassBalanceBatchERM
from spuco.models import model_factory
from spuco.evaluate import Evaluator

classes = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
difficulty = SpuriousFeatureDifficulty.MAGNITUDE_LARGE

trainset = SpuCoMNIST(
    root="/data/mnist/",
    spurious_feature_difficulty=difficulty,
    spurious_correlation_strength=0.995,
    classes=classes,
    split="train"
)
trainset.initialize()

testset = SpuCoMNIST(
    root="/data/mnist/",
    spurious_feature_difficulty=difficulty,
    classes=classes,
    split="test"
)
testset.initialize()

model = model_factory("lenet", trainset[0][0].shape, trainset.num_classes)

erm = ERM(
    model=model,
    num_epochs=1,
    trainset=trainset,
    batch_size=64,
    optimizer=SGD(model.parameters(), lr=1e-2, momentum=0.9, nesterov=True),
    verbose=True
)
erm.train()

evaluator = Evaluator(
    testset=trainset,
    group_partition=testset.group_partition,
    group_weights=trainset.group_weights,
    batch_size=64,
    model=model,
    verbose=True
)
evaluator.evaluate()

evaluator = Evaluator(
    testset=testset,
    group_partition=testset.group_partition,
    group_weights=trainset.group_weights,
    batch_size=64,
    model=model,
    verbose=True
)
evaluator.evaluate()

cluster = Cluster(
    Z=erm.trainer.get_trainset_outputs(),
    class_labels=trainset.labels,
    cluster_alg=ClusterAlg.KMEANS,
    num_clusters=2,
    verbose=True
)
group_partition = cluster.infer_groups()

model = model_factory("lenet", trainset[0][0].shape, trainset.num_classes)
group_balance_erm = GroupBalanceBatchERM(
    model=model,
    num_epochs=5,
    trainset=trainset,
    group_partition=group_partition,
    batch_size=64,
    optimizer=SGD(model.parameters(), lr=1e-3, weight_decay=5e-4, momentum=0.9, nesterov=True),
    verbose=True
)
group_balance_erm.train()

evaluator = Evaluator(
    testset=testset,
    group_partition=testset.group_partition,
    group_weights=trainset.group_weights,
    batch_size=64,
    model=model,
    verbose=True
)
evaluator.evaluate()

evaluator = Evaluator(
    testset=trainset,
    group_partition=testset.group_partition,
    group_weights=trainset.group_weights,
    batch_size=64,
    model=model,
    verbose=True
)
evaluator.evaluate()

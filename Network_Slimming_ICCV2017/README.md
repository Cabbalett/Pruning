# Channel-Wise Pruning 

Realization of the paper;
[Learning Efficient Convolutional Networks through Network Slimming](https://arxiv.org/abs/1708.06519)

1. Install required library:

        pip install -r requirements.txt

2. Train Vgg model with the training objective introduced in the paper:

        python train.py

3. Finetune the pruned model(75% pruned)

        python finetune.py

more detailed explanation in [My Blog](https://cabbalett.github.io/week17/Week17-1/)

vgg model based on [huyvnphan's github](https://github.com/huyvnphan/PyTorch_CIFAR10)

Article to implement: [transformers](https://arxiv.org/pdf/2010.11929.pdf) \
Original article about transformers: [original](https://arxiv.org/pdf/1706.03762.pdf)

Build docker container:

- ```docker build -t transformers . ```
- ```docker run -it --name transformers transformers```

Запускать:

- train ```python train_model.py --dataset mnist```
- test  ```python test_model.py --dataset mnist --image-path xxx --weights-path xxx```

Available options:

- ```--dataset - mnist or fruit```
- ```--image-path - sample_images/fruits.jpg or sample_images/mnist.jpg```
- ```--weights-path - weights/fruit.ckpt or weights/mnist.ckpt```
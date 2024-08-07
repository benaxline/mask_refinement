# mask_refinement
Refining masks using [MGMatting](https://github.com/yucornetto/MGMatting) model. The goal of this project was to evaluate if the MGMatting model would be a viable option for refining masks. I was looking to see if the model would work for different types of images since it was trained primarily on two datasets. I evaluated the model that was trained on [real-world portraits](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/qyu13_jh_edu/Edl8x0nQjy1JhGP6rcV0N-cB654HpmZZa5bwW9rYUvmsJg?e=J3lSba).

## Use case examples

1. Given this image and rough mask of the dog and cat:

![dogs](https://github.com/benaxline/mask_refinement/blob/main/data/image/pet/download.jpg) ![dogs](https://github.com/benaxline/mask_refinement/blob/main/data/masks/pet/download.png)

We can produce a better mask by running them through the MGMatting model to get this mask: 

![refined](https://github.com/benaxline/mask_refinement/blob/main/data/refined/pet/download.png)


2. Given this image and rough mask of nature:

![tree](https://github.com/benaxline/mask_refinement/blob/main/data/image/nature/nature-2.jpg) ![tree](https://github.com/benaxline/mask_refinement/blob/main/data/masks/nature/nature-2.png)

We can produce a better mask of the tree by running them through the MGMatting model:

![tree](https://github.com/benaxline/mask_refinement/blob/main/data/refined/nature/nature-2.png)

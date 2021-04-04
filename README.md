# StyleSpace Analysis: Disentangled Controls for StyleGAN Image Generation 

Demo video: <a href="https://youtu.be/U7qRotRGr1w"><img src="https://img.shields.io/badge/-YouTube-red?&style=for-the-badge&logo=youtube&logoColor=white" height=20></a>
CVPR 2021 Oral: <a href="(https://arxiv.org/abs/2011.12799"><img src="https://www.google.co.il/imgres?imgurl=https%3A%2F%2Fupload.wikimedia.org%2Fwikipedia%2Fcommons%2Fthumb%2Fa%2Fa8%2FArXiv_web.svg%2F1200px-ArXiv_web.svg.png&imgrefurl=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FArXiv&tbnid=Tn0wrYL0AnL_VM&vet=12ahUKEwjErbDDtOXvAhXayrsIHcxvDPYQMygAegUIARDFAQ..i&docid=rmcHwZXSenBHBM&w=1200&h=648&q=arxiv&ved=2ahUKEwjErbDDtOXvAhXayrsIHcxvDPYQMygAegUIARDFAQ" height=20></a>
Single Channel Manipulation: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/betterze/StyleSpace/blob/main/StyleSpace_single.ipynb)



![](imgs/disentanglement.png)

> **StyleSpace Analysis: Disentangled Controls for StyleGAN Image Generation**<br>
Zongze Wu, Dani Lischinski, Eli Shechtman <br>
> [paper](https://arxiv.org/abs/2011.12799) (CVPR 2021 Oral) 
> [video](https://youtu.be/U7qRotRGr1w)
>
>**Abstract:** We explore and analyze the latent style space of StyleGAN2, a state-of-the-art architecture for image generation, using models pretrained on several different datasets. We first show that StyleSpace, the space of channel-wise style parameters, is significantly more disentangled than the other intermediate latent spaces explored by previous works. Next, we describe a method for discovering a large collection of style channels, each of which is shown to control a distinct visual attribute in a highly localized and disentangled manner. Third, we propose a simple method for identifying style channels that control a specific attribute, using a pretrained classifier or a small number of example images. Manipulation of visual attributes via these StyleSpace controls is shown to be better disentangled than via those proposed in previous works. To show this, we make use of a newly proposed Attribute Dependency metric. Finally, we demonstrate the applicability of StyleSpace controls to the manipulation of real images. Our findings pave the way to semantically meaningful and well-disentangled image manipulations via simple and intuitive interfaces.



#### generated face manipulation
![](imgs/ffhq.png)

#### generated car and bedroom manipulation
![](imgs/car_bed.png)
#### real face manipulation
![](imgs/real.png)




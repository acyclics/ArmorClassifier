# Introduction
Using simulated images to train CNNs is a very attractive idea due to the amount of data needed for training CNNs prior to it generalizing well. However, a problem with using simulated images, perhaps from software such as Unity, is a generalization-gap between images from simulation and images in real life.

# Solution
One method of tackling a deep learning problem is to combat it with another deep learning problem. GANs are neural networks that learns to generate images via a generator and a discriminator. If we have some data on how our generated images should actually look like (perhaps a real image), we can train a GAN to modify Unity generated images to become closer in pixel-distribution to what a real-image would look like, where feedback on the effectiveness of the generator would be given by the discriminator, which serves as a signal for the generator to improve.

# Examples
Below is a Unity generated image depicting an armor:

![](https://github.com/acyclics/ArmorClassifier/blob/master/examples/unity_demo.png)

Below are GAN modified images to better match the actual distribution:

![](https://github.com/acyclics/ArmorClassifier/blob/master/examples/gen1.png) ![](https://github.com/acyclics/ArmorClassifier/blob/master/examples/gen2.png)
![](https://github.com/acyclics/ArmorClassifier/blob/master/examples/gen3.png) ![](https://github.com/acyclics/ArmorClassifier/blob/master/examples/gen4.png)

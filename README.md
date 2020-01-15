# affine-brisk-registration
The combination of affine transformation and brisk registration algorithm is suitable for quick registration in different scenarios.

This registration algorithm can be applied to:
1. Same object, different perspectives, rollover, change, etc.
2. Different environments and perspectives for the same object.
3. Different objects, same shape.

Registration speed:
1. Two 1980*1080 pictures are about 1.5s
2. Two 1280*720 pictures are about 0.7s

Different Angle registration of the same picture

![Image text](https://raw.githubusercontent.com/Zz-ww/affine-brisk-registration/master/1_1_out.jpg)

Part of the graph is registered with this image

![Image text](https://raw.githubusercontent.com/Zz-ww/affine-brisk-registration/master/1_2_out.jpg)

Different environment registration for the same object

![Image text](https://raw.githubusercontent.com/Zz-ww/affine-brisk-registration/master/2_1_out.jpg)

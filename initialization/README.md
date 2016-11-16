# Initialization

This directory contains initialization helpers.

## LSUV initialization

The `lsuv_init.py` is taken from the [LSUVinit] project and uses the layer-sequential unit variance approach to initialize the network with efficient weights for ReLU layers.

Usage:

```bash
python lsuv_init.py solver.prototxt initialized_weights.caffemodel $ALGORITHM $MODE
```

where `$ALGORITHM` is one of `LSUV`, `Orthonormal` or `OrthonormalLSUV` and `$MODE` is either `cpu` or `gpu`. GPU operation is much faster but RAM limited and LSUV initialization likes to have really large batch sizes.

The weights can then be used in "fine-tuning" mode with

```bash
caffe train -solver solver.prototxt -weights initialized_weights.caffemodel
```

### License

LSUVInit is copyrighted by the University of California. See `lsuv_init.LICENSE.txt` for further information.

```
All contributions by the University of California:
Copyright (c) 2014, 2015, The Regents of the University of California (Regents)
All rights reserved.
```

[LSUVinit]: https://github.com/ducha-aiki/LSUVinit
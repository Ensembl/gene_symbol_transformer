#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Dimitrios Paraschas (dimitrios@ebi.ac.uk)


"""
Neural network pipeline.
"""


# standard library imports

# third party imports
import torch

# project imports


def main():
    """
    main function
    """
    x = torch.rand(5, 3)
    print(x)


if __name__ == "__main__":
    main()

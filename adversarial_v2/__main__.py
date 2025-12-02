#!/usr/bin/env python3
"""
Entry point for running adversarial_v2 as a module.

Usage:
    python -m adversarial_v2 --mode static --num-cycles 10
"""
from .cotrain import main

if __name__ == "__main__":
    main()

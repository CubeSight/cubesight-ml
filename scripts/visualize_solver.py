"""
CLI script to launch the CubeSight-ML 3D Rubik's Cube Solver Studio in a desktop window.
Allows running Beam Search inference interactively for individual stages or the full 4-stage CFOP pipeline.
"""

from cubesight_ml.visualization.server import main

if __name__ == "__main__":
    main()

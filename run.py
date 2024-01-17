
import os

from training.trainer import Trainer


def main():
    hparams = "ethz_usstof_win.json"
    trainer = Trainer(hparams_file=hparams)
    trainer.train()
    trainer.evaluate()

if __name__ == "__main__":
    main()

"""
TODO:
- general:
    - add cascade to occupancy grid

- speed up algroithm:
    - resolve TODO: optimize

- improve algorithm:
    - debug memory leak
    - make PSO optimization

- implement real time running:
    - implement data accumulation
    - remove all dependencies from step or time

- evaluation
    -evaluate at robot centre


- investigate why keep_pixels_in_angle_range cannot be [-5,5]



"""

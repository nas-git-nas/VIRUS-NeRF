import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

from training.trainer import Trainer


def main():
    hparams = "ethz_usstof_gpu.json"
    trainer = Trainer(hparams_file=hparams)
    trainer.train()
    trainer.evaluate()


# def evaluate():
#     chkt_path = "results/robot_at_home/20231016_1119"
#     trainer = Trainer(hparams_file="rh_gpu.json")
#     trainer.loadCheckpoint(ckpt_path=os.path.join(chkt_path, "model.pth"))
#     trainer.evaluate()


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


- investigate why keep_pixels_in_angle_range cannot be [-5,5]



"""

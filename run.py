import os

from training.trainer import Trainer


def main():
    trainer = Trainer(hparams_file="rh_gpu.json")
    trainer.train()
    trainer.evaluate()


def evaluate():
    chkt_path = "results/robot_at_home/20231016_1119"
    trainer = Trainer(hparams_file="rh_gpu.json")
    trainer.loadCheckpoint(ckpt_path=os.path.join(chkt_path, "model.pth"))
    trainer.evaluate()

if __name__ == "__main__":
    main()
    # evaluate()
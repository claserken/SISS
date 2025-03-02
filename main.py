import hydra
from omegaconf import DictConfig, OmegaConf
from abc import ABC, abstractmethod
import datetime
import pytz
import os
import uuid

class Task(ABC):
    @abstractmethod
    def run(self):
        pass

@hydra.main(version_base=None, config_path="config")
def main(cfg: DictConfig) -> None:

    # Print config
    # print(OmegaConf.to_yaml(cfg))

    if cfg.resume_from_checkpoint:
        cfg.output_dir += f"/{os.path.dirname(cfg.resume_from_checkpoint)}"
    else:
        pacific_timezone = pytz.timezone('US/Pacific')
        current_datetime_pt = datetime.datetime.now(tz=pacific_timezone)
        formatted_datetime = current_datetime_pt.strftime("%Y-%m-%d_%H-%M-%S")
        dir_id = str(uuid.uuid4()) # prevent directory name conflicts in case of running multiple experiments in parallel
        cfg.output_dir += f"/{formatted_datetime}_{dir_id}"
    
    # Instantiate and run task
    task = hydra.utils.instantiate(
        cfg.task,
        cfg=cfg,
        _recursive_=False
    )
    task.run()


if __name__ == "__main__":
    main()

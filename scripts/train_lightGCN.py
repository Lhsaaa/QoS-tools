import sys
import argparse
import warnings
from pathlib import Path
from logging import getLogger
from recbole.utils import init_seed, set_color


sys.path.append(str(Path(__file__).parent.parent))

from config.configuration import Config
from utils.logger import init_logger
from data.dataset import GeneralGraphDataset
from data.utils import data_reparation
from utils.utils import get_flops, get_model
from trainer.trainer import Trainer


warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()

parser.add_argument(
    "--dataset", "-d", type=str, default="wsdream-rt", help="name of datasets"
)
parser.add_argument(
    "--model", "-m", type=str, default="LightGCN", help="name of models"
)
parser.add_argument(
    "--train_batch_size", type=int, default=512, help="batch size"
)
parser.add_argument(
    "--learning_rate", type=float, default=0.005, help="learning rate"
)
parser.add_argument(
    "--weight_decay", type=float, default=1e-07, help="weight decay"
)
parser.add_argument(
    "--n_layers", type=int, default=4, help="n layers"
)
parser.add_argument(
    "--split_ratio", type=float, default = 0.025, help="Density"
)


args, _ = parser.parse_known_args()

config = Config(model=args.model, dataset=args.dataset)

# update config
config["train_batch_size"] = args.train_batch_size
config["learning_rate"] = args.learning_rate
# config["weight_decay"] = args.weight_decay
config["n_layers"] = args.n_layers
config["split_ratio"] = args.split_ratio

init_logger(config)
init_seed(config["seed"], True)

logger = getLogger()
logger.info(config)

dataset = GeneralGraphDataset(config)
train_data, test_data = data_reparation(config, dataset)

# 使用训练集的数据建图
model = get_model(config["model"])(config, train_data.dataset).to(config["device"])
logger.info(model)

flops = get_flops(model, dataset, config["device"], logger)
logger.info(set_color("FLOPs", "blue") + f": {flops}")

trainer = Trainer(config, model)
best_valid_score, best_valid_result = trainer.fit(
    train_data, test_data, saved=True, show_progress=bool(config["show_progress"]))

logger.info(set_color("best valid ", "yellow") + f": {best_valid_result}")
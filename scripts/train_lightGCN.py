import argparse
import warnings
from logging import getLogger
from recbole.utils import init_seed, set_color


from config.configuration import Config
from utils.logger import init_logger
from data.dataset import GeneralGraphDataset
from data.utils import data_reparation
from utils.utils import get_flops, get_model


warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()

parser.add_argument(
    "--dataset", "-d", type=str, default="wsdream-rt", help="name of datasets"
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


args, _ = parser.parse_known_args()

config = Config(model=args.model, dataset=args.dataset)
config["train_batch_size"], config["learning_rate"], config["weight_decay"], config["n_layers"] = args.train_batch_size, args.learning_rate, args.weight_decay, args.n_layers

init_logger(config)
init_seed(config["seed"], True)

logger = getLogger()
logger.info(config)

dataset = GeneralGraphDataset(config)
train_data, test_data = data_reparation(config, dataset)

# 必须传入train_data, 必须使用训练集的数据建图
model = get_model(config["model"])(config, train_data.dataset).to(config["device"])
logger.info(model)

flops = get_flops(model, dataset, config["device"], logger)
logger.info(set_color("FLOPs", "blue") + f": {flops}")

trainer = Trainer(config, model)
best_valid_score, best_valid_result = trainer.fit(
    train_data, test_data, saved=True, show_progress=bool(config["show_progress"]))

logger.info(set_color("best valid ", "yellow") + f": {best_valid_result}")
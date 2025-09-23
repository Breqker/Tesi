import time
import os
import torch
from conf.attack_parser import attack_parse_args
from conf.recommend_parser import recommend_parse_args
from util.DataLoader import DataLoader
from util.tool import seedSet
from ARLib import ARLib
import json
import random

# PATCH per Python >=3.13: random.sample accetta set o dict
orig_sample = random.sample
def sample_patch(population, k):
    if not isinstance(population, (list, tuple)):
        population = list(population)
    return orig_sample(population, k)
random.sample = sample_patch

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Patch .cuda() per usare device corretto
orig_tensor_cuda = torch.Tensor.cuda
def tensor_cuda_patch(self, *args, **kwargs):
    return self.to(device)
torch.Tensor.cuda = tensor_cuda_patch

# Numero di thread su CPU
torch.set_num_threads(os.cpu_count())
print(f"Number of CPU threads: {torch.get_num_threads()}")

# Funzione per controllare dataset
def check_dataset_files(data_path, dataset_name):
    required_files = ["train.txt", "val.txt", "test.txt"]
    full_path = os.path.join(data_path, dataset_name)
    print(f"Checking dataset folder: {full_path}")
    missing_files = []
    for f in required_files:
        path = os.path.join(full_path, f)
        if not os.path.exists(path):
            missing_files.append(path)
        else:
            print(f"Found: {path}")
    if missing_files:
        raise FileNotFoundError(f"Missing files: {missing_files}")
    return full_path

class Main():
    # 1. Load configuration
    recommend_args = recommend_parse_args()
    attack_args = attack_parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = recommend_args.gpu_id
    seed = recommend_args.seed
    seedSet(seed)

    # Controllo dataset
    dataset_folder = check_dataset_files(recommend_args.data_path, recommend_args.dataset)

    # 2. Import recommend model and attack model
    import_str = 'from recommender.' + recommend_args.model_name + ' import ' + recommend_args.model_name
    exec(import_str)
    import_str = 'from attack.' + attack_args.attackCategory + "." + attack_args.attackModelName + ' import ' + attack_args.attackModelName
    exec(import_str)

    # 3. Load data
    data = DataLoader(recommend_args)

    # 4. Define recommend model and attack model, and define ARLib to control the process
    recommend_model = eval(recommend_args.model_name)(recommend_args, data)
    attack_model = eval(attack_args.attackModelName)(attack_args, data)
    arlib = ARLib(recommend_model, attack_model, recommend_args, attack_args)

    s = time.time()
    metrics_log = {"clean": [], "attacks": {}}

    # 5. Train and test in clean data
    print("=== CLEAN RUN ===")
    arlib.RecommendTrain()
    arlib.RecommendTest()
    if hasattr(recommend_model, "bestPerformance") and len(recommend_model.bestPerformance) > 1:
        metrics_log["clean"].append(recommend_model.bestPerformance[1])

    # 6. Attack
    arlib.PoisonDataAttack()
    for step in range(arlib.times):
        print(f"=== Attack step {step} ===")
        arlib.RecommendTrain(attack=step)
        arlib.RecommendTest(attack=step)
        if hasattr(recommend_model, "bestPerformance") and len(recommend_model.bestPerformance) > 1:
            metrics_log["attacks"][f"attack_{step}"] = recommend_model.bestPerformance[1]

    e = time.time()
    print("Running time: %f s" % (e - s))

    # Stampa e salva metriche
    print("\n=== Metriche raccolte ===")
    print(json.dumps(metrics_log, indent=4))
    with open("../metrics_log.json", "w") as f:
        json.dump(metrics_log, f, indent=4)
    print("Saved metrics to metrics_log.json")

if __name__ == '__main__':
    Main()

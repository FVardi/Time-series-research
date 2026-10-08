"""TF-C (Zhang et al., 2022), official code, wrapped (D10 mode A, D19, D36, D37).

The official code lives unmodified in external/tfc (git submodule, commit 9667582,
the January 2023 version in code/TFC; D36). It is used for transfer: pretrain on
the source data (FD-A), then fine-tune encoder + classifier on the target training
set (FD-B) and test on the target test set.

The official main.py and Trainer() are scripts that read files from fixed paths, so
they are not called. Instead, the steps of main.py and of Trainer() are written out
below, calling the official functions for everything inside an epoch
(Load_Dataset, TFC, target_classifier, model_pretrain, model_finetune, model_test).
Differences from running main.py, all from D37 or needed to run:
  - the debugging subset is switched off (main.py hard-codes subset=True; D37);
  - after every fine-tuning epoch the model is also scored on the target
    validation set, which the official code never loads (D37);
  - the "best" fine-tuned model is kept in memory instead of in a file under
    experiments_logs/ (same behaviour);
  - pretraining and fine-tuning run in one process; each phase is seeded exactly as
    main.py seeds its own process, and fine-tuning starts from a fresh model that
    loads the pretrained weights, as main.py does.
"""

import copy

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier

from harness.methods.external import import_official

# Fix needed to run with NumPy >= 1.24 (D19: fixes go here, the official files stay unedited):
# the official code uses np.float, which NumPy removed, in two 'except' branches
# (when a batch lacks a class and the AUROC cannot be computed). It meant Python's float.
if not hasattr(np, "float"):
    np.float = float

CODE = "tfc/code/TFC"  # the official 2023 code inside the submodule
NAMES = ("model", "trainer", "dataloader", "augmentations", "loss", "utils")
trainer = import_official(CODE, "trainer", local_names=NAMES)  # also brings model.py and loss.py
dataloader = import_official(CODE, "dataloader", local_names=NAMES)  # also brings augmentations.py
Config = import_official("tfc/code", "config_files.FD_A_Configs", local_names=("config_files",)).Config


def official_config(s):
    """The official FD-A config (main.py picks the config of the pretraining dataset).

    Every setting we rely on is written in our YAML (s['official_config']) and must
    equal the official file; a difference stops the run, so nothing is silently changed.
    """
    configs = Config()
    for name, value in s["official_config"].items():
        obj = configs
        for part in name.split(".")[:-1]:  # e.g. 'Context_Cont.temperature'
            obj = getattr(obj, part)
        official = getattr(obj, name.split(".")[-1])
        if official != value:
            raise ValueError(f"{name}: YAML says {value}, official config says {official}")
    return configs


def seed_like_main(seed):
    """Exactly the seeding at the top of the official main.py."""
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def as_official(X, y):
    """Our arrays -> the dict the official Load_Dataset expects.

    X: (n, time, 1) float64 from the canonical reader; y: (n,) class labels 0..2.
    The official files hold 'samples' (n, 1, time) float64 and 'labels' (n,) int64.
    """
    return {"samples": torch.from_numpy(np.transpose(X, (0, 2, 1)).copy()),
            "labels": torch.from_numpy(np.asarray(y, dtype=np.int64))}


def loaders(data, configs, mode, subset):
    """The official data_generator(), written out: (pretraining, fine-tuning, test) loaders.

    data: dict with 'source_train', 'target_train', 'target_test' (and 'target_val'), each (X, y).
    Same order, sizes and loader settings as the official function.
    """
    D = dataloader.Load_Dataset
    train = D(as_official(*data["source_train"]), configs, mode, target_dataset_size=configs.batch_size, subset=subset)
    finetune = D(as_official(*data["target_train"]), configs, mode, target_dataset_size=configs.target_batch_size, subset=subset)
    test = D(as_official(*data["target_test"]), configs, mode, target_dataset_size=configs.target_batch_size, subset=False)

    L = torch.utils.data.DataLoader
    return (L(train, batch_size=configs.batch_size, shuffle=True, drop_last=configs.drop_last, num_workers=0),
            L(finetune, batch_size=configs.target_batch_size, shuffle=True, drop_last=configs.drop_last, num_workers=0),
            L(test, batch_size=configs.target_batch_size, shuffle=True, drop_last=False, num_workers=0))


def val_loader(data, configs, mode):
    """Added (D37): the target validation set, with the test loader's settings.

    Load_Dataset shuffles with NumPy's random-number generator; its state is saved and
    restored so that the official part of the run is unchanged by this step.
    """
    state = np.random.get_state()
    val = dataloader.Load_Dataset(as_official(*data["target_val"]), configs, mode,
                                  target_dataset_size=configs.target_batch_size, subset=False)
    np.random.set_state(state)
    return torch.utils.data.DataLoader(val, batch_size=configs.target_batch_size, shuffle=True, drop_last=False, num_workers=0)


def new_model_and_optimizers(configs, device, s):
    """As in main.py: TF-C model, classifier, and an Adam optimizer for each (weight decay hard-coded there)."""
    model = trainer.TFC(configs).to(device)
    classifier = trainer.target_classifier(configs).to(device)
    adam = dict(lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=s["weight_decay"])
    return model, classifier, torch.optim.Adam(model.parameters(), **adam), torch.optim.Adam(classifier.parameters(), **adam)


def pretrain(data, s, device, seed):
    """main.py with --training_mode pre_train.

    Returns the pretrained weights, the loss per epoch, and whether training diverged.
    If an epoch's loss is not finite (NaN/inf), training stops there (D44): the weights
    are then invalid and cannot recover, so the remaining epochs would change nothing.
    """
    configs = official_config(s)
    seed_like_main(seed)
    train_dl, _, _ = loaders(data, configs, "pre_train", s["subset"])
    model, _, model_opt, _ = new_model_and_optimizers(configs, device, s)

    criterion = torch.nn.CrossEntropyLoss()  # passed to model_pretrain, as in Trainer()
    losses = []
    for epoch in range(1, configs.num_epoch + 1):
        loss = trainer.model_pretrain(model, model_opt, criterion, train_dl, configs, device, "pre_train")
        losses.append({"epoch": epoch, "loss": float(loss)})
        if not np.isfinite(float(loss)):
            print(f"  pretraining diverged at epoch {epoch} (loss {float(loss)}); stopping this seed", flush=True)
            return model.state_dict(), pd.DataFrame(losses), True
    return model.state_dict(), pd.DataFrame(losses), False


def finetune_and_test(pretrained, data, s, device, seed):
    """main.py with --training_mode fine_tune_test, plus validation scores (D37).

    Returns one row per fine-tuning epoch with the fine-tuning loss, and the
    scores on the target test and validation sets (fractions, not percent).
    """
    configs = official_config(s)
    mode = "fine_tune_test"
    seed_like_main(seed)
    _, finetune_dl, test_dl = loaders(data, configs, mode, s["subset"])
    val_dl = val_loader(data, configs, mode)
    model, classifier, model_opt, clf_opt = new_model_and_optimizers(configs, device, s)
    model.load_state_dict(pretrained)  # main.py loads ckp_last.pt here

    # The fine-tuning loop of the official Trainer(), written out.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_opt, "min")
    names = ["accuracy", "precision", "recall", "f1", "auroc", "auprc"]  # order of model_test's 'performance'
    rows, total_f1, best = [], [], None
    for epoch in range(1, configs.num_epoch + 1):
        ft_loss, emb_ft, label_ft, ft_f1 = trainer.model_finetune(
            model, model_opt, finetune_dl, configs, device, mode, classifier=classifier, classifier_optimizer=clf_opt)
        scheduler.step(ft_loss)

        # Official behaviour: keep the model with the best fine-tuning F1 so far, load it
        # back before testing, and continue training from it in the next epoch.
        if len(total_f1) == 0 or ft_f1 > max(total_f1):
            best = (copy.deepcopy(model.state_dict()), copy.deepcopy(classifier.state_dict()))
        total_f1.append(ft_f1)
        model.load_state_dict(best[0])
        classifier.load_state_dict(best[1])

        _, _, _, _, emb_test, label_test, test_perf = trainer.model_test(
            model, test_dl, configs, device, mode, classifier=classifier, classifier_optimizer=clf_opt)
        # Added (D37): the same scoring on the target validation set. Iterating a data
        # loader draws from PyTorch's random-number generator, so its state is saved and
        # restored: the official part of the run stays exactly as without this step.
        rng = torch.get_rng_state()
        _, _, _, _, _, _, val_perf = trainer.model_test(
            model, val_dl, configs, device, mode, classifier=classifier, classifier_optimizer=clf_opt)
        torch.set_rng_state(rng)

        # The official KNN classifier (k = 5) on the fine-tuning embeddings, as in Trainer().
        knn = KNeighborsClassifier(n_neighbors=5).fit(emb_ft, label_ft)
        knn_pred = knn.predict(emb_test.detach().cpu().numpy())

        row = {"epoch": epoch, "finetune_loss": float(ft_loss), "finetune_f1": float(ft_f1)}
        row.update({f"test_{n}": float(v) / 100 for n, v in zip(names, test_perf)})
        row.update({f"val_{n}": float(v) / 100 for n, v in zip(names, val_perf)})
        row["knn_test_accuracy"] = accuracy_score(label_test, knn_pred)
        row["knn_test_f1"] = f1_score(label_test, knn_pred, average="macro")
        rows.append(row)
    return pd.DataFrame(rows)

from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np

def plot_learning_curves(models, X_train, y_train, dataset_name, results_dir):
    fig, axes = plt.subplots(1, len(models), figsize=(18, 5))
    fig.suptitle(f"Learning Curves — {dataset_name}")

    for ax, (model_name, model) in zip(axes, models.items()):
        train_sizes, train_scores, val_scores = learning_curve(
            model, X_train, y_train,
            cv=3,
            scoring="roc_auc",
            train_sizes=np.linspace(0.1, 1.0, 10),
            n_jobs=-1
        )
        ax.plot(train_sizes, train_scores.mean(axis=1), label="Train")
        ax.plot(train_sizes, val_scores.mean(axis=1),   label="Val")
        ax.fill_between(train_sizes,
            train_scores.mean(axis=1) - train_scores.std(axis=1),
            train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.1)
        ax.fill_between(train_sizes,
            val_scores.mean(axis=1) - val_scores.std(axis=1),
            val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.1)
        ax.set_title(model_name)
        ax.set_xlabel("Training samples")
        ax.set_ylabel("AUC")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plot_dir = results_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_dir / "learning_curves.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved to {plot_dir / 'learning_curves.png'}")


def plot_mlp_loss(model, results_dir):
    if not hasattr(model, "loss_curve_"):
        print("No loss curve available")
        return

    plt.figure(figsize=(8, 5))
    plt.plot(model.loss_curve_, label="Train loss")
    if hasattr(model, "validation_scores_"):
        plt.plot(model.validation_scores_, label="Val score")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("MLP Loss Curve")
    plt.legend()
    plt.grid(True)

    plot_dir = results_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_dir / "mlp_loss_curve.png", dpi=150, bbox_inches="tight")
    plt.show()
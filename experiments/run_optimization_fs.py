import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score
import time
import json
from pathlib import Path
import random
import warnings
warnings.filterwarnings('ignore')

from src.utils.paths import SPLITS_DIR, MODELS_DIR, FEATURES_DIR, COMPARISONS_DIR
from src.utils.data_io import load_splits

# Datasets and models
DATASETS = ["ibtracs.last3years"]
MODELS = {
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "xgboost": XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss')
}

# Target number of features
TARGET_N_FEATURES = [10, 20, 30]

def evaluate_feature_subset(X_train, y_train, feature_indices):
    """
    Evaluate a feature subset using cross-validation.
    Returns average F1 score.
    """
    if len(feature_indices) == 0:
        return 0.0
    
    X_subset = X_train.iloc[:, feature_indices]
    
    # Use a fast model for evaluation
    model = LogisticRegression(max_iter=1000, random_state=42)
    
    # 3-fold CV for speed
    scores = cross_val_score(model, X_subset, y_train, cv=3, scoring='f1', n_jobs=-1)
    
    return scores.mean()


def pso_feature_selection(X_train, y_train, n_features, n_particles=30, iterations=20):
    """
    Particle Swarm Optimization for feature selection.
    
    Parameters:
    - n_particles: Number of particles in swarm
    - iterations: Number of optimization iterations
    """
    print(f"  Running PSO: particles={n_particles}, iterations={iterations}")
    
    n_total_features = X_train.shape[1]
    
    # Initialize particles: continuous values in [0, 1]
    particles = np.random.rand(n_particles, n_total_features)
    velocities = np.random.rand(n_particles, n_total_features) * 0.1
    
    # Personal best positions and scores
    personal_best_positions = particles.copy()
    personal_best_scores = np.zeros(n_particles)
    
    # Global best
    global_best_position = None
    global_best_score = 0.0
    
    # PSO parameters
    w = 0.7  # Inertia weight
    c1 = 1.5  # Cognitive parameter
    c2 = 1.5  # Social parameter
    
    for iteration in range(iterations):
        for i in range(n_particles):
            # Convert continuous position to binary feature selection
            # Select top n_features based on particle position values
            feature_indices = np.argsort(particles[i])[-n_features:].tolist()
            
            # Evaluate fitness
            fitness = evaluate_feature_subset(X_train, y_train, feature_indices)
            
            # Update personal best
            if fitness > personal_best_scores[i]:
                personal_best_scores[i] = fitness
                personal_best_positions[i] = particles[i].copy()
            
            # Update global best
            if fitness > global_best_score:
                global_best_score = fitness
                global_best_position = particles[i].copy()
        
        if iteration % 5 == 0:
            print(f"    Iteration {iteration}: Best F1 = {global_best_score:.4f}")
        
        # Update velocities and positions
        for i in range(n_particles):
            r1 = np.random.rand(n_total_features)
            r2 = np.random.rand(n_total_features)
            
            velocities[i] = (w * velocities[i] +
                           c1 * r1 * (personal_best_positions[i] - particles[i]) +
                           c2 * r2 * (global_best_position - particles[i]))
            
            particles[i] = particles[i] + velocities[i]
            particles[i] = np.clip(particles[i], 0, 1)  # Keep in bounds
    
    # Get final best features
    selected_features = np.argsort(global_best_position)[-n_features:].tolist()
    
    print(f"    Final best F1: {global_best_score:.4f}")
    
    return selected_features


def differential_evolution_fs(X_train, y_train, n_features, population_size=30, generations=20):
    """
    Differential Evolution for feature selection.
    
    Parameters:
    - population_size: Number of candidate solutions
    - generations: Number of evolution iterations
    """
    print(f"  Running DE: population={population_size}, generations={generations}")
    
    n_total_features = X_train.shape[1]
    
    # DE parameters
    F = 0.8   # Differential weight (mutation factor)
    CR = 0.9  # Crossover probability
    
    # Initialize population: continuous values in [0, 1]
    population = np.random.rand(population_size, n_total_features)
    
    # Evaluate initial population
    fitness = np.zeros(population_size)
    for i in range(population_size):
        feature_indices = np.argsort(population[i])[-n_features:].tolist()
        fitness[i] = evaluate_feature_subset(X_train, y_train, feature_indices)
    
    best_idx = np.argmax(fitness)
    best_solution = population[best_idx].copy()
    best_fitness = fitness[best_idx]
    
    for generation in range(generations):
        for i in range(population_size):
            # Mutation: Select 3 random distinct individuals
            indices = [idx for idx in range(population_size) if idx != i]
            a, b, c = random.sample(indices, 3)
            
            # Create mutant vector
            mutant = population[a] + F * (population[b] - population[c])
            mutant = np.clip(mutant, 0, 1)
            
            # Crossover
            trial = np.copy(population[i])
            crossover_mask = np.random.rand(n_total_features) < CR
            trial[crossover_mask] = mutant[crossover_mask]
            
            # Ensure at least one parameter comes from mutant
            if not crossover_mask.any():
                random_idx = random.randint(0, n_total_features - 1)
                trial[random_idx] = mutant[random_idx]
            
            # Selection: Compare trial with current
            trial_features = np.argsort(trial)[-n_features:].tolist()
            trial_fitness = evaluate_feature_subset(X_train, y_train, trial_features)
            
            # Replace if better
            if trial_fitness > fitness[i]:
                population[i] = trial
                fitness[i] = trial_fitness
                
                # Update global best
                if trial_fitness > best_fitness:
                    best_fitness = trial_fitness
                    best_solution = trial.copy()
        
        if generation % 5 == 0:
            print(f"    Generation {generation}: Best F1 = {best_fitness:.4f}")
    
    # Get final best features
    selected_features = np.argsort(best_solution)[-n_features:].tolist()
    
    print(f"    Final best F1: {best_fitness:.4f}")
    
    return selected_features


def train_with_selected_features(X_train, X_test, y_train, y_test,
                                  method_name, n_features, dataset_name, selected_features):
    """Train all models with selected features and return results."""
    
    results = []
    
    X_train_selected = X_train.iloc[:, selected_features]
    X_test_selected = X_test.iloc[:, selected_features]
    
    for model_name, model in MODELS.items():
        # Train
        start_time = time.time()
        model.fit(X_train_selected, y_train)
        train_time = time.time() - start_time
        
        # Predict
        y_pred = model.predict(X_test_selected)
        y_pred_proba = model.predict_proba(X_test_selected)[:, 1]
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        # Save model
        model_dir = MODELS_DIR / "optimization_fs" / dataset_name / method_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        import pickle
        model_filename = f"{model_name}_k{n_features}.pkl"
        with open(model_dir / model_filename, "wb") as f:
            pickle.dump(model, f)
        
        # Store results
        results.append({
            "dataset": dataset_name,
            "model": model_name,
            "method": method_name,
            "n_features": n_features,
            "accuracy": accuracy,
            "f1_score": f1,
            "auc": auc,
            "train_time": train_time
        })
    
    return results


def run_optimization_fs(dataset_name):
    """Run optimization-based feature selection methods."""
    
    print(f"\n{'='*60}")
    print(f"Optimization-Based Feature Selection: {dataset_name}")
    print(f"{'='*60}\n")
    
    # Load data
    splits = load_splits(dataset_name)
    X_train = splits["X_train"]
    X_test = splits["X_test"]
    y_train = splits["y_train"]
    y_test = splits["y_test"]
    
    print(f"Original features: {X_train.shape[1]}")
    print(f"Training samples: {len(X_train)}\n")
    
    all_results = []
    
    # ==================== PARTICLE SWARM OPTIMIZATION ====================
    print(f"\n{'='*60}")
    print("PARTICLE SWARM OPTIMIZATION (PSO)")
    print(f"{'='*60}\n")
    
    for n_features in TARGET_N_FEATURES:
        if n_features >= X_train.shape[1]:
            continue
        
        print(f"\n--- Selecting {n_features} features ---")
        
        start_time = time.time()
        selected_indices = pso_feature_selection(X_train, y_train, n_features)
        selection_time = time.time() - start_time
        
        selected_features = X_train.columns[selected_indices].tolist()
        
        print(f"  Selection completed in {selection_time:.2f}s")
        
        # Save selected features
        features_dir = FEATURES_DIR / dataset_name / "pso"
        features_dir.mkdir(parents=True, exist_ok=True)
        
        with open(features_dir / f"selected_k{n_features}.json", "w") as f:
            json.dump({
                "n_features": n_features,
                "selected_features": selected_features,
                "selection_time": selection_time
            }, f, indent=2)
        
        # Train models and collect results
        results = train_with_selected_features(
            X_train, X_test, y_train, y_test,
            "pso", n_features, dataset_name, selected_indices
        )
        
        all_results.extend(results)
        
        # Print summary
        for r in results:
            print(f"  {r['model']:20s} - F1: {r['f1_score']:.4f}, AUC: {r['auc']:.4f}")
    
    # ==================== DIFFERENTIAL EVOLUTION ====================
    print(f"\n{'='*60}")
    print("DIFFERENTIAL EVOLUTION (DE)")
    print(f"{'='*60}\n")
    
    for n_features in TARGET_N_FEATURES:
        if n_features >= X_train.shape[1]:
            continue
        
        print(f"\n--- Selecting {n_features} features ---")
        
        start_time = time.time()
        selected_indices = differential_evolution_fs(X_train, y_train, n_features)
        selection_time = time.time() - start_time
        
        selected_features = X_train.columns[selected_indices].tolist()
        
        print(f"  Selection completed in {selection_time:.2f}s")
        
        # Save selected features
        features_dir = FEATURES_DIR / dataset_name / "differential_evolution"
        features_dir.mkdir(parents=True, exist_ok=True)
        
        with open(features_dir / f"selected_k{n_features}.json", "w") as f:
            json.dump({
                "n_features": n_features,
                "selected_features": selected_features,
                "selection_time": selection_time
            }, f, indent=2)
        
        # Train models and collect results
        results = train_with_selected_features(
            X_train, X_test, y_train, y_test,
            "differential_evolution", n_features, dataset_name, selected_indices
        )
        
        all_results.extend(results)
        
        # Print summary
        for r in results:
            print(f"  {r['model']:20s} - F1: {r['f1_score']:.4f}, AUC: {r['auc']:.4f}")
    
    # Save all results
    results_dir = COMPARISONS_DIR / "tables"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(results_dir / f"optimization_fs_{dataset_name}.csv", index=False)
    
    print(f"\n{'='*60}")
    print(f"✓ Optimization FS complete for {dataset_name}")
    print(f"✓ Saved results to {results_dir / f'optimization_fs_{dataset_name}.csv'}")
    print(f"{'='*60}\n")
    
    return results_df


if __name__ == "__main__":
    all_results = []
    
    for ds in DATASETS:
        results = run_optimization_fs(ds)
        all_results.append(results)
    
    # Combine results
    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(COMPARISONS_DIR / "tables" / "optimization_fs_all.csv", index=False)
    
    print("\n" + "="*60)
    print("ALL OPTIMIZATION FEATURE SELECTION COMPLETE")
    print("="*60)
    print(f"\nPSO Results:")
    print(combined[combined['method'] == 'pso'].groupby('n_features')['f1_score'].mean().to_string())
    print(f"\nDifferential Evolution Results:")
    print(combined[combined['method'] == 'differential_evolution'].groupby('n_features')['f1_score'].mean().to_string())
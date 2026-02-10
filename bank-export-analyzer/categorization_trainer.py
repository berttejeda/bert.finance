import pandas as pd
import yaml
import os
import re
from collections import defaultdict

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

class CategorizationTrainer:
    def __init__(self, df, config_file="config.yaml"):
        self.df = df
        self.config_file = config_file
        self.vectorizer = None
        self.kmeans = None
        self.new_mappings = defaultdict(list)

    def cluster_descriptions(self, n_clusters=None):
        """
        Clusters transaction descriptions using TF-IDF and K-Means.
        If n_clusters is None, it defaults to len(df) // 10 or 5, whichever is smaller.
        """
        if not SKLEARN_AVAILABLE:
            print("Error: scikit-learn is required for training. Please install it using 'pip install scikit-learn'.")
            return {}

        descriptions = self.df['Description'].unique().tolist()
        
        if len(descriptions) < 5:
            print("Not enough unique descriptions to cluster. Skipping clustering.")
            return {desc: i for i, desc in enumerate(descriptions)}

        if n_clusters is None:
             n_clusters = max(2, min(len(descriptions) // 5, 15)) # Heuristic for cluster count

        print(f"Vectorizing {len(descriptions)} unique descriptions...")
        self.vectorizer = TfidfVectorizer(stop_words='english')
        X = self.vectorizer.fit_transform(descriptions)

        print(f"Clustering into {n_clusters} clusters...")
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.kmeans.fit(X)

        # Map description to cluster label
        cluster_map = {desc: label for desc, label in zip(descriptions, self.kmeans.labels_)}
        return cluster_map

    def interactive_labeling(self):
        """
        Iterates through clusters and asks the user for a category label.
        """
        if self.df.empty:
            print("No transactions to train on.")
            return

        cluster_map = self.cluster_descriptions()
        if not cluster_map:
            return

        # Group descriptions by cluster
        clusters = defaultdict(list)
        for desc, label in cluster_map.items():
            clusters[label].append(desc)

        print("\n--- Interactive Categorization Training ---")
        print("I will show you groups of similar transactions. You can Assign a category name, or Skip.")
        
        updated_count = 0
        
        for label, items in clusters.items():
            print(f"\nCluster {label + 1} / {len(clusters)}:")
            print("Sample descriptions:")
            for item in items[:5]: # Show top 5 examples
                print(f"  - {item}")
            if len(items) > 5:
                print(f"  ... and {len(items) - 5} more")

            choice = input(f"Enter Category Name (or press Enter to skip): ").strip()
            
            if choice:
                # Ask for a keyword pattern to capture this
                print(f"Great! Assigning to '{choice}'.")
                pattern = input("Enter a unique keyword/regex to match these (e.g., 'uber'): ").strip()
                
                if pattern:
                    self.new_mappings[choice].append(pattern)
                    updated_count += 1
                    print(f"Added pattern '{pattern}' to category '{choice}'.")
                else:
                    print("No pattern provided. Skipping.")
            else:
                print("Skipping.")

        if updated_count > 0:
            self.save_new_mappings()
        else:
            print("\nNo new categories added.")

    def save_new_mappings(self):
        """
        Appends the new mappings to the YAML config file.
        """
        print(f"\nSaving new categories to {self.config_file}...")
        
        current_config = {}
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                current_config = yaml.safe_load(f) or {}

        # Merge new mappings
        # Use 'categories' key if present, otherwise create it (enforcing new structure)
        # We assume the user wants to migrate/use the new structure
        if 'categories' not in current_config:
            # If the config seems to be the old flat format (no 'categories' key but has data)
            # we might be careful, but the instruction is to use 'categories' key.
            # We'll just ensure the key exists.
            current_config['categories'] = {}
            
        target_categories = current_config['categories']

        for category, patterns in self.new_mappings.items():
            if category not in target_categories:
                target_categories[category] = []
            
            # Avoid duplicates
            existing_patterns = set(target_categories[category])
            for p in patterns:
                if p not in existing_patterns:
                    target_categories[category].append(p)
        
        # Write back to file
        with open(self.config_file, 'w') as f:
            yaml.dump(current_config, f, sort_keys=True)
        
        print("Configuration updated successfully!")

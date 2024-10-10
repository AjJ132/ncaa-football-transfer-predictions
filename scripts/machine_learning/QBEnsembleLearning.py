import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix, 
    roc_curve, 
    auc, 
    precision_recall_curve, 
    f1_score
)
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import os
import matplotlib.pyplot as plt
import seaborn as sns
import json


class QBEnsembleLearning:
    def __init__(self, data_path, predictions_path, visualizations_path, stats_path):
        self.data_path = data_path
        self.predictions_path = predictions_path
        self.visualizations_path = visualizations_path
        self.stats_path = stats_path
        self.data = None
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None
        self.models = []
        self.preprocessor = None
        self.class_weights = None
        self.model_stats = {}

    def load_data(self):
        self.data = pd.read_csv(self.data_path)
        print("Data loaded successfully.")
        print(f"Shape of the data: {self.data.shape}")

    def preprocess_data(self):
        # Convert 'transfer' to boolean
        self.data['transfer'] = self.data['transfer'].astype(bool)

        # Select features for prediction
        numeric_features = ['g', 'passing_att', 'passing_comp', 'passing_pct.', 'passing_yards',
                            'passing_yards/att', 'passing_td', 'passing_int', 'passing_rating',
                            'passing_att/g', 'passing_yards/g', 'rushing_att', 'rushing_yards',
                            'rushing_avg.', 'rushing_td', 'rushing_att/g', 'rushing_yards/g']
        categorical_features = ['yr']

        # Create preprocessing steps
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
            ('onehot', OneHotEncoder(drop='first', sparse_output=False))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        # Split data into features (X) and target (y)
        X = self.data[numeric_features + categorical_features]
        y = self.data['transfer']

        # Split data into training (2016-2022) and validation (2023) sets
        mask = self.data['season'] < 2023
        self.X_train, self.X_val = X[mask], X[~mask]
        self.y_train, self.y_val = y[mask], y[~mask]

        # Store validation set index for later use
        self.val_index = self.data[~mask].index

        # Fit and transform the training data, transform the validation data
        self.X_train = self.preprocessor.fit_transform(self.X_train)
        self.X_val = self.preprocessor.transform(self.X_val)

        # Apply SMOTE to the training data
        smote = SMOTE(random_state=42)
        self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)

        # Get feature names after preprocessing
        numeric_features_transformed = numeric_features
        categorical_features_transformed = self.preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features).tolist()
        self.feature_names = numeric_features_transformed + categorical_features_transformed

        # Compute class weights (not necessary after SMOTE, but kept for consistency)
        self.class_weights = compute_class_weight('balanced', classes=np.unique(self.y_train), y=self.y_train)
        self.class_weights = dict(zip(np.unique(self.y_train), self.class_weights))

        print("Data preprocessed successfully.")
        print(f"Training set shape after SMOTE: {self.X_train.shape}")
        print(f"Validation set shape: {self.X_val.shape}")
        print(f"Class distribution after SMOTE: {np.bincount(self.y_train.astype(int))}")
    
    def train_models(self):
        # Initialize and train multiple models
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        lr_model = LogisticRegression(random_state=42, max_iter=1000)

        models = [rf_model, gb_model, lr_model]
        model_names = ['Random Forest', 'Gradient Boosting', 'Logistic Regression']

        for model, name in zip(models, model_names):
            model.fit(self.X_train, self.y_train)
            self.models.append((name, model))
            print(f"{name} trained successfully.")

    def evaluate_models(self):
        for name, model in self.models:
            y_pred = model.predict(self.X_val)
            y_pred_proba = model.predict_proba(self.X_val)[:, 1]

            # Calculate metrics
            accuracy = accuracy_score(self.y_val, y_pred)
            f1 = f1_score(self.y_val, y_pred)
            confusion_mat = confusion_matrix(self.y_val, y_pred)
            classification_rep = classification_report(self.y_val, y_pred, output_dict=True)

            # ROC curve
            fpr, tpr, _ = roc_curve(self.y_val, y_pred_proba)
            roc_auc = auc(fpr, tpr)

            # Precision-Recall curve
            precision, recall, _ = precision_recall_curve(self.y_val, y_pred_proba)
            pr_auc = auc(recall, precision)

            # Store metrics
            self.model_stats[name] = {
                'accuracy': accuracy,
                'f1_score': f1,
                'confusion_matrix': confusion_mat.tolist(),
                'classification_report': {
                    'accuracy': classification_rep['accuracy'],
                    'macro avg': classification_rep['macro avg'],
                    'weighted avg': classification_rep['weighted avg']
                },
                'roc_auc': roc_auc,
                'pr_auc': pr_auc
            }

            # Print summary
            print(f"\n{name} Validation Accuracy: {accuracy:.4f}")
            print(f"{name} F1 Score: {f1:.4f}")
            print(f"\n{name} Confusion Matrix:")
            print(confusion_mat)

            # Plot ROC curve
            self.plot_curve(fpr, tpr, roc_auc, name, 'ROC', 'False Positive Rate', 'True Positive Rate')

            # Plot Precision-Recall curve
            self.plot_curve(recall, precision, pr_auc, name, 'Precision-Recall', 'Recall', 'Precision')

        # Generate feature importance plot for Random Forest
        self.plot_feature_importance()

        return self.models[0][1].predict(self.X_val)  # Return predictions from the first model (Random Forest)

    def plot_curve(self, x, y, auc_score, model_name, curve_type, xlabel, ylabel):
        plt.figure()
        plt.plot(x, y, color='darkorange', lw=2, label=f'{curve_type} curve (AUC = {auc_score:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(f'{model_name} {curve_type} Curve')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(self.visualizations_path, f'{model_name.lower().replace(" ", "_")}_{curve_type.lower()}_curve.png'))
        plt.close()

    def plot_feature_importance(self):
        rf_model = self.models[0][1]  # Assuming Random Forest is the first model
        feature_importance = rf_model.feature_importances_
        feature_importance = 100.0 * (feature_importance / feature_importance.max())
        sorted_idx = np.argsort(feature_importance)
        pos = np.arange(sorted_idx.shape[0]) + .5

        plt.figure(figsize=(12, 6))
        plt.barh(pos, feature_importance[sorted_idx], align='center')
        plt.yticks(pos, [self.feature_names[i] for i in sorted_idx])
        plt.xlabel('Relative Importance')
        plt.title('Variable Importance (Random Forest)')
        plt.tight_layout()
        plt.savefig(os.path.join(self.visualizations_path, 'feature_importance.png'))
        plt.close()


    def save_predictions(self, y_pred):
        # Create a DataFrame with predictions
        predictions_df = pd.DataFrame({
            'name': self.data.loc[self.val_index, 'name'],
            'team_name': self.data.loc[self.val_index, 'team_name'],
            'season': self.data.loc[self.val_index, 'season'],
            'actual_transfer': self.y_val,
            'predicted_transfer': y_pred
        })

        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.predictions_path), exist_ok=True)

        # Save predictions to CSV
        predictions_df.to_csv(self.predictions_path, index=False)
        print(f"Predictions saved to {self.predictions_path}")

        # Generate a heatmap of misclassifications
        misclassified = predictions_df[predictions_df['actual_transfer'] != predictions_df['predicted_transfer']]
        pivot = pd.pivot_table(misclassified, values='name', index='team_name', columns='actual_transfer', aggfunc='count', fill_value=0)
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot, annot=True, fmt='d', cmap='YlOrRd')
        plt.title('Misclassifications by Team and Actual Transfer Status')
        plt.tight_layout()
        plt.savefig(os.path.join(self.visualizations_path, 'misclassifications_heatmap.png'))
        plt.close()

    def save_model_stats(self):
        with open(self.stats_path, 'w') as f:
            json.dump(self.model_stats, f, indent=4)
        print(f"Model statistics saved to {self.stats_path}")

    def run(self):
        self.load_data()
        self.preprocess_data()
        self.train_models()
        y_pred = self.evaluate_models()
        self.save_predictions(y_pred)
        self.save_model_stats()


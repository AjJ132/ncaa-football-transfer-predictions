import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix, 
    roc_auc_score,
    roc_curve, 
    auc, 
    precision_recall_curve, 
    f1_score
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
import os
import json

class QBTimeSeriesApproach:
    def __init__(self, data_path, output_path):
        self.data_path = data_path
        self.output_path = output_path
        self.data = None
        self.models = []
        self.predictions = []

    def load_data(self):
        self.data = pd.read_csv(self.data_path)
        print("Data loaded successfully.")

    def apply_smote(self, X, y):
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        return X_resampled, y_resampled

    def preprocess_data(self):
        features = [
            'g', 'g_diff', 'passing_rating', 'pass_rating_perc', 'other_qbs',
            'attempts_per_game', 'playing_time_ratio', 'team_win_pct', 'team_win_pct_change',
            'depth_chart_position', 'passing_rating_yoy_change', 'passing_att_yoy_change',
            'passing_yards_yoy_change', 'passing_td_yoy_change', 'offense_plays', 'offense_total_yards',
            'offense_yards/play', 'offense_yards/g', 'offense_plays_yoy_change', 'offense_total_yards_yoy_change', 'offense_yards/play_yoy_change', 'offense_yards/g_yoy_change'
        ]
        
        self.train_data = self.data[self.data['season'].isin(range(2016, 2023))]
        self.validation_data = self.data[self.data['season'] == 2023]

        self.X_train = self.train_data[features]
        self.y_train = self.train_data['transfer']
        self.X_val = self.validation_data[features]
        self.y_val = self.validation_data['transfer']

        scaler = StandardScaler()
        self.X_train_scaled = pd.DataFrame(scaler.fit_transform(self.X_train), 
                                        columns=self.X_train.columns, 
                                        index=self.X_train.index)
        self.X_val_scaled = pd.DataFrame(scaler.transform(self.X_val), 
                                        columns=self.X_val.columns, 
                                        index=self.X_val.index)

        print("Data preprocessed successfully.")

    def train_models(self, n_models=10):
       model_types = [
           RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
           GradientBoostingClassifier(n_estimators=100, random_state=42),
           LogisticRegression(class_weight='balanced', random_state=42),
           SVC(probability=True, class_weight='balanced', random_state=42),
           XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=len(self.y_train[self.y_train==0])/len(self.y_train[self.y_train==1]), random_state=42)
       ]

       self.feature_importances = np.zeros(len(self.X_train_scaled.columns))
       
       for i in range(n_models):
           for model in model_types:
               model.fit(self.X_train_scaled, self.y_train)
               self.models.append(model)
               
               if hasattr(model, 'feature_importances_'):
                   self.feature_importances += model.feature_importances_
               elif hasattr(model, 'coef_'):
                   self.feature_importances += np.abs(model.coef_[0])

       self.feature_importances /= (n_models * len(model_types))
       print(f"{n_models * len(model_types)} models trained successfully with adjusted class weights.")

    def make_predictions(self):
        for model in self.models:
            self.predictions.append(model.predict_proba(self.X_val_scaled)[:, 1])

        self.final_predictions = np.mean(self.predictions, axis=0)
        print("Predictions made successfully.")

    def find_optimal_threshold(self):
       fpr, tpr, thresholds = roc_curve(self.y_val, self.final_predictions)
       optimal_idx = np.argmax(tpr - fpr)
       optimal_threshold = thresholds[optimal_idx]
       return optimal_threshold

    def evaluate_model(self):
        optimal_threshold = self.find_optimal_threshold()
        y_pred = (self.final_predictions > optimal_threshold).astype(int)
        
        self.accuracy = accuracy_score(self.y_val, y_pred)
        self.roc_auc = roc_auc_score(self.y_val, self.final_predictions)
        self.classification_rep = classification_report(self.y_val, y_pred, 
                                                        output_dict=True, 
                                                        zero_division=0)

        print(f"Model evaluated successfully with optimal threshold: {optimal_threshold:.4f}")

    def create_output_folder(self):
        base_path = os.path.join(self.output_path, 'TimeSeriesApproach')
        os.makedirs(base_path, exist_ok=True)
        
        existing_folders = [f for f in os.listdir(base_path) if f.isdigit()]
        if existing_folders:
            latest_num = max(map(int, existing_folders))
            new_folder_num = latest_num + 1
        else:
            new_folder_num = 1
        
        self.output_folder = os.path.join(base_path, str(new_folder_num))
        os.makedirs(self.output_folder, exist_ok=True)
        print(f"Output folder created: {self.output_folder}")

    def save_results(self):
        predictions_df = pd.DataFrame({
            'name': self.validation_data['name'],
            'season': self.validation_data['season'],
            'actual': self.y_val,
            'predicted_prob': self.final_predictions,
            'predicted_class': (self.final_predictions > 0.5).astype(int)
        })
        predictions_df.to_csv(os.path.join(self.output_folder, 'predictions_2023.csv'), index=False)

        with open(os.path.join(self.output_folder, 'model_performance.json'), 'w') as f:
            json.dump({
                'accuracy': self.accuracy,
                'roc_auc': self.roc_auc,
                'classification_report': self.classification_rep
            }, f, indent=4)

        fpr, tpr, _ = roc_curve(self.y_val, self.final_predictions)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {self.roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()
        plt.savefig(os.path.join(self.output_folder, 'roc_curve.png'))
        plt.close()

        cm = confusion_matrix(self.y_val, (self.final_predictions > 0.5).astype(int))
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(self.output_folder, 'confusion_matrix.png'))
        plt.close()

        feature_importance_df = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': self.feature_importances
        }).sort_values('importance', ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=feature_importance_df)
        plt.title('Feature Importance')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_folder, 'feature_importance.png'))
        plt.close()

        print("Results and feature importance saved successfully.")

    def run(self):
        self.load_data()
        self.preprocess_data()
        self.train_models()
        self.make_predictions()
        self.evaluate_model()
        self.create_output_folder()
        self.save_results()
        print("Ensemble model training and evaluation completed successfully using SMOTE for class balancing.")
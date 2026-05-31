import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.image import imread
import numpy as np

LOGS_DIR = os.path.join(os.path.dirname(__file__), 'logs', 'my_logs')

def display_image(title, img_path, figsize=(10, 6)):
    if not os.path.exists(img_path):
        print(f"[SKIP] {img_path} not found")
        return
    plt.figure(figsize=figsize)
    img = imread(img_path)
    plt.imshow(img)
    plt.axis('off')
    plt.title(title, fontsize=14, pad=10)
    plt.tight_layout()
    plt.show()

def show_baseline_models():
    print("\n" + "="*60)
    print("PHASE 1: BASELINE — 4 Models")
    print("="*60)

    for model_name in ['Facenet_mtcnn', 'Facenet_retinaface', 'Facenet512_mtcnn', 'Facenet512_retinaface']:
        model_dir = os.path.join(LOGS_DIR, model_name)
        print(f"\n--- {model_name} ---")

        display_image(f"{model_name} — Accuracy per Epoch",
                      os.path.join(model_dir, 'accuracy.png'))
        display_image(f"{model_name} — Loss per Epoch",
                      os.path.join(model_dir, 'loss.png'))
        display_image(f"{model_name} — FAR vs FRR",
                      os.path.join(model_dir, 'far_frr.png'))

    display_image("Comparison — 4 Baseline Models",
                  os.path.join(LOGS_DIR, 'comparison_4models.png'), figsize=(14, 8))

    csv_path = os.path.join(LOGS_DIR, 'MLP_Model_Evaluation_Metrics.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print("\n--- Summary Table ---")
        print(df.to_string(index=False))

def show_improved_models():
    print("\n" + "="*60)
    print("PHASE 2: IMPROVED Facenet512")
    print("="*60)

    for model_name in ['Facenet512_mtcnn_concat_MLP_v4', 'Facenet512_retinaface_concat_MLP_v1']:
        model_dir = os.path.join(LOGS_DIR, model_name)
        print(f"\n--- {model_name} ---")

        display_image(f"{model_name} — Training Curves",
                      os.path.join(model_dir, 'training_curves.png'))

    display_image("Comparison — Facenet512 Improved",
                  os.path.join(LOGS_DIR, 'comparison_facenet512_improved.png'), figsize=(14, 8))

    csv_path = os.path.join(LOGS_DIR, 'MLP_Facenet512_Improved_Metrics.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print("\n--- Summary Table ---")
        print(df.to_string(index=False))

def show_regularized_models():
    print("\n" + "="*60)
    print("PHASE 3: REGULARIZED Facenet512")
    print("="*60)

    full_models = [
        'Facenet512_mtcnn_Reg1_512x256x128_regularized',
        'Facenet512_retinaface_Reg1_512x256x128_regularized'
    ]
    for model_name in full_models:
        model_dir = os.path.join(LOGS_DIR, model_name)
        print(f"\n--- {model_name} ---")

        display_image(f"{model_name} — Training Curves",
                      os.path.join(model_dir, 'training_curves.png'))
        display_image(f"{model_name} — Confusion Matrix",
                      os.path.join(model_dir, 'confusion_matrix.png'))

        report_path = os.path.join(model_dir, 'final_report.txt')
        if os.path.exists(report_path):
            with open(report_path, 'r', encoding='utf-8') as f:
                print(f.read())

    display_image("Comparison — Facenet512 Regularized",
                  os.path.join(LOGS_DIR, 'comparison_facenet512_regularized.png'), figsize=(14, 8))

    csv_path = os.path.join(LOGS_DIR, 'MLP_Facenet512_Regularized_CV.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print("\n--- K-Fold CV Summary ---")
        print(df.to_string(index=False))

def show_final_roc():
    print("\n" + "="*60)
    print("FINAL ROC CURVE")
    print("="*60)
    display_image("ROC Curve — All Models",
                  os.path.join(LOGS_DIR, 'ROC_Curve_Final.png'), figsize=(12, 8))

def show_overall_comparison():
    print("\n" + "="*60)
    print("OVERALL COMPARISON")
    print("="*60)

    data = {
        'Phase': ['Baseline', 'Improved', 'Regularized*', 'Regularized*'],
        'Model': ['Facenet_retinaface', 'Facenet512_retinaface_v1',
                  'Facenet512_retinaface_Reg1', 'Facenet512_retinaface_Reg3'],
        'Test Acc': [0.9408, 0.9592, 0.9642, 0.9625],
        'F1': [0.9407, 0.9582, 0.9637, 0.9621],
        'EER': [0.0567, 0.0508, 0.0399, 0.0384],
        'Overfit Gap': [0.0367, 0.0323, 0.0306, 0.0330],
        'Infer': ['0.18ms', '0.44ms', '0.24ms', '0.24ms']
    }
    df = pd.DataFrame(data)
    print(df.to_string(index=False))
    print("\n* K-Fold CV average")

if __name__ == '__main__':
    show_baseline_models()
    show_improved_models()
    show_regularized_models()
    show_final_roc()
    show_overall_comparison()
    print("\n=== DONE ===")

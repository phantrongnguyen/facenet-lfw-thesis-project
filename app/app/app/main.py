import matplotlib.pyplot as plt

from src.dataset_loader import load_pairs
from src.evaluator import evaluate
from src.visualization import plot_roc
from src.reporting import create_report_run
from src.reporting import build_results_dataframe
from src.reporting import save_report_files

from src.config import DATASET_PATH
from src.config import PAIRS_FILE
from src.config import MODELS
from src.config import THRESHOLD
from src.config import REPORTS_DIR


pairs=load_pairs(PAIRS_FILE,DATASET_PATH)
report_context=create_report_run(REPORTS_DIR)

results={}

plt.figure()

for model in MODELS:

    similarities,labels,acc,FAR,FRR,time=evaluate(
        model,
        pairs,
        THRESHOLD[model]
    )

    results[model]=(acc,FAR,FRR,time)

    plot_roc(labels,similarities,model)

plt.plot([0,1],[0,1],'--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")

plt.legend()

roc_path=report_context["run_dir"] / "roc_curve.png"

plt.savefig(roc_path,dpi=300,bbox_inches="tight")

plt.show()

results_df=build_results_dataframe(results,THRESHOLD)

report_files=save_report_files(
    report_context,
    results_df,
    dataset_path=DATASET_PATH,
    pairs_file=PAIRS_FILE,
    total_pairs=len(pairs)
)


print("\n===== FINAL RESULTS =====\n")

for model in results:

    acc,FAR,FRR,time=results[model]

    print(model)
    print("Accuracy:",acc)
    print("FAR:",FAR)
    print("FRR:",FRR)
    print("Inference Time:",time)
    print()

print("Reports saved:")
print("Run folder:",report_context["run_dir"])
print("CSV:",report_files["csv"])
print("JSON:",report_files["json"])
print("Markdown:",report_files["markdown"])
print("Metadata:",report_files["metadata"])
print("ROC:",roc_path)

# ====== LƯU 2 MODELS ======
print("\n===== SAVING MODELS =====")
import os
from deepface import DeepFace

SAVE_DIR = report_context["run_dir"] / "saved_models"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

for model in MODELS:
    print(f"Đang tải và lưu mô hình {model}...")
    tf_model = DeepFace.build_model(model)
    model_path = os.path.join(SAVE_DIR, f"{model.lower()}_weights.h5")
    # Lấy đối tượng tf/keras model thực sự ở bên trong wrapper thông qua .model
    tf_model.model.save(model_path)
    print(f"✅ Đã lưu {model} tại: {model_path}")
print("Hoàn tất!")
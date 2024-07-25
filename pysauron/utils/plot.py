import subprocess
import sys

try:
    import matplotlib.pyplot as plt
except:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'matplotlib'])
    import matplotlib.pyplot as plt



def plot_anomaly_score(video_name:str, gt_labels:list, pred_labels:list, log_dir:str) -> None: 
    """
       Plot GT anomaly score and predicted

    Args:
        video_name (str): video filename
        gt_labels (list): list of GT labels
        pred_labels (list): list of predicted labels
        log_dir (str): directory where to save the plot
    """    
    save_folder = log_dir / 'plots' 
    save_folder.mkdir(parents=True, exist_ok=True)
    save_path = save_folder / f"{video_name.replace('.mp4', '')}_anomaly_score.png"
    print("Saving plot to: ", str(save_path))

    plt.figure(figsize=(16, 4))
    plt.fill_between(list(range(len(gt_labels))), gt_labels, 
        color='r', alpha=0.3, edgecolor='None')
    if pred_labels:
        plt.plot(pred_labels)
    plt.title(f"{video_name.replace('.mp4', '')} anomaly scores")
    plt.ylim(0., 1.0)
    plt.xlim(0, len(gt_labels))
    plt.xlabel('Frames')
    plt.ylabel('Anomaly score')
    plt.grid(True)
    plt.savefig(str(save_path))
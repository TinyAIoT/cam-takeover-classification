import matplotlib.pyplot as plt
import numpy as np

# all frequencies calculated using the rgb 1x16 overtake model and the rgb 48x192 surface model
data = {
    # takeover-and-surface
    "All separate threads": {
        "singlecore": {"camera capture": 12.44, "overtake classification": 4.18, "surface classification": 2.27},
        "multicore": {"camera capture": 12.44,  "overtake classification": 2.61, "surface classification": 2.35},
    },
    # takeover-and-surface-semi-concurrent
    "camera capture thread \nand classification thread": {
        "singlecore": {"camera capture": 12.5, "both classifications": 4.97},
        "multicore": {"camera capture": 12.5, "both classifications": 4.98},
    },
    # takeover-and-surface-concurrent
    "All concurrent": {
        "singlecore": {"All": 4.76},
        "multicore": {"All": 5.01},
    }
}

# ---------------------------------------
# PREPARE PLOTTING
# ---------------------------------------

use_cases = list(data.keys())
settings = list(next(iter(data.values())).keys())

thread_categories = sorted(
    {t for use_case in data.values() for setting in use_case.values() for t in setting.keys()}
)

labels = []
plot_matrix = {t: [] for t in thread_categories}

for use_case in use_cases:
    for setting in settings:
        labels.append(f"{use_case}\n{setting}")

        threads = data[use_case][setting]
        for t in thread_categories:
            plot_matrix[t].append(threads.get(t, 0))

# ---------------------------------------
# CREATE GROUPED BAR CHART
# ---------------------------------------

x = np.arange(len(labels))
num_threads = len(thread_categories)
width = 0.8 / num_threads

fig, ax = plt.subplots(figsize=(14, 7))

for i, (t, values) in enumerate(plot_matrix.items()):
    ax.bar(x + i * width, values, width, label=t)

# ---------------------------------------
# ADD HORIZONTAL LINES
# ---------------------------------------

min_freq = 1.5
max_freq = 8

ax.axhline(min_freq, color='red', linestyle='--', linewidth=1.5,
           label='minimum frequency in trainingsdata')
ax.axhline(max_freq, color='green', linestyle='--', linewidth=1.5,
           label='maximum frequency in trainingsdata')

# ---------------------------------------
# LABELS, TITLE, LEGEND
# ---------------------------------------

ax.set_ylabel('Frequency (Hz)')
ax.set_title('Thread Frequencies Under Different Use Cases and Settings')
ax.set_xticks(x + width * num_threads / 2)
ax.set_xticklabels(labels, rotation=45, ha='right')

# To avoid duplicate legend entries when adding lines
handles, labels_legend = ax.get_legend_handles_labels()
unique = list(dict(zip(labels_legend, handles)).items())
ax.legend([u[1] for u in unique], [u[0] for u in unique], title="Legend")

ax.grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.show()
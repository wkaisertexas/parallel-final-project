import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io

# 2. Read the data
df = pd.read_csv('grid_search.csv')

# Set global style
sns.set_style("whitegrid")

# Plot 1: Time vs Tile Size (grouped by Block Size)
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='Tile Size', y='Time', hue='Block Size', marker='o', palette='viridis', legend='full')
plt.xscale('log', base=2)
plt.yscale('log')
plt.title('Execution Time vs Tile Size (grouped by Block Size)')
plt.xlabel('Tile Size')
plt.ylabel('Time (s) - Log Scale')
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.tight_layout()
plt.savefig('imgs/time_vs_tile_size.png')
print("Saved time_vs_tile_size.png")

# Plot 2: Time vs Block Size (grouped by Tile Size)
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='Block Size', y='Time', hue='Tile Size', marker='o', palette='magma', legend='full')
plt.xscale('log', base=2)
plt.yscale('log')
plt.title('Execution Time vs Block Size (grouped by Tile Size)')
plt.xlabel('Block Size')
plt.ylabel('Time (s) - Log Scale')
plt.legend(title='Tile Size', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.tight_layout()
plt.savefig('imgs/time_vs_block_size.png')
print("Saved time_vs_block_size.png")

# Plot 3: Heatmap of Time
plt.figure(figsize=(10, 8))
pivot_table = df.pivot(index="Block Size", columns="Tile Size", values="Time")
sns.heatmap(pivot_table, annot=True, fmt=".4f", cmap="YlGnBu_r", cbar_kws={'label': 'Time (s)'})
plt.title('Heatmap of Execution Time (Block Size vs Tile Size)')
plt.tight_layout()
plt.savefig('imgs/heatmap_time.png')
print("Saved heatmap_time.png")

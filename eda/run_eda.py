import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageStat
import numpy as np
from collections import Counter
import random
import cv2
import pandas as pd
from sklearn.cluster import KMeans


def _single_axis_palette_plot(plot_fn, data, x, y, palette, **kwargs):
    ax = plot_fn(
        data=data,
        x=x,
        y=y,
        hue=x,
        palette=palette,
        dodge=False,
        **kwargs,
    )
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()
    return ax


def compute_image_stats(img_path):
    try:
        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        stat = ImageStat.Stat(img)
        mean_rgb = stat.mean
        std_rgb = stat.stddev
        
        # Open CV for Laplacian (blurriness/edges)
        cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        laplacian_var = cv2.Laplacian(cv_img, cv2.CV_64F).var()
        
        file_size_kb = os.path.getsize(img_path) / 1024.0
        
        # Resize small for average pixel computation (to save memory)
        img_small = np.array(img.resize((64, 64)))
        
        return {
            'width': w,
            'height': h,
            'aspect_ratio': w / h if h > 0 else 0,
            'mean_r': mean_rgb[0],
            'mean_g': mean_rgb[1],
            'mean_b': mean_rgb[2],
            'std_r': std_rgb[0],
            'std_g': std_rgb[1],
            'std_b': std_rgb[2],
            'laplacian': laplacian_var,
            'file_size_kb': file_size_kb,
            'img_small': img_small
        }
    except Exception as e:
        print(f"Error computing stats for {img_path}: {e}")
        return None

def create_eda_plots(data_dir, output_dir):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    splits = ['train', 'val']
    
    # Store aggregated info for cross-split comparison
    split_counts = {}
    
    sns.set_theme(style="whitegrid")
    
    for split in splits:
        split_dir = data_dir / split
        if not split_dir.exists():
            print(f"Directory {split_dir} not found. Skipping...")
            continue
            
        classes = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])
        counts = {}
        sample_images = {}
        
        all_stats = []
        
        print(f"\n[{split.upper()}] Scanning images and extracting deep statistics...")
        
        for cls in classes:
            cls_dir = split_dir / cls
            images = [p for p in cls_dir.glob("*.*") if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']]
            if images:
                counts[cls] = len(images)
                sample_images[cls] = random.choice(images)
                
                # Sample up to 300 images per class for deep stats to save time & memory
                sampled_for_stats = random.sample(images, min(300, len(images)))
                for img_path in sampled_for_stats:
                    s = compute_image_stats(img_path)
                    if s:
                        s['class'] = cls
                        s['path'] = str(img_path)
                        all_stats.append(s)
                
        split_counts[split] = counts
                
        if not counts or not all_stats:
            print(f"No valid images found in {split_dir} for EDA. Skipping...")
            continue
            
        df = pd.DataFrame(all_stats)
        
        # Make a specific output folder per split so it's clean
        split_out = output_dir / split
        split_out.mkdir(exist_ok=True)
        
        print(f"[{split.upper()}] Generating 10+ advanced multi-dimensional visual analyses...")
        
        # ==========================================
        # 1. Bar plot & 2. Pie Chart (Class Distrib)
        # ==========================================
        counts_df = pd.DataFrame({
            'class': list(counts.keys()),
            'count': list(counts.values()),
        })
        plt.figure(figsize=(10, 6))
        _single_axis_palette_plot(
            sns.barplot,
            data=counts_df,
            x='class',
            y='count',
            palette='viridis',
        )
        plt.title(f'01 - Class Count Distribution - {split.upper()} Split', fontsize=14)
        plt.xlabel('Classes')
        plt.ylabel('Number of Images')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(split_out / '01_class_distribution_bar.png', dpi=200)
        plt.close()
        
        plt.figure(figsize=(8, 8))
        plt.pie(counts.values(), labels=counts.keys(), autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
        plt.title(f'02 - Class Proportion - {split.upper()} Split', fontsize=14)
        plt.tight_layout()
        plt.savefig(split_out / '02_class_pie_chart.png', dpi=200)
        plt.close()
        
        # ==========================================
        # 3. Sample grids
        # ==========================================
        num_classes = len(sample_images)
        cols = min(5, num_classes)
        rows = max(1, (num_classes + cols - 1) // cols)
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
        if rows == 1 and cols == 1:
            axes = np.array([axes])
        axes = np.atleast_1d(axes).ravel()
        
        for i, (cls, img_path) in enumerate(sample_images.items()):
            try:
                img = Image.open(img_path).convert('RGB')
                axes[i].imshow(img)
                axes[i].set_title(f"{cls}\n(n={counts[cls]})", fontsize=12)
                axes[i].axis('off')
            except Exception as e:
                pass
                
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
            
        plt.tight_layout()
        plt.savefig(split_out / '03_random_sample_grid.png', dpi=200)
        plt.close()

        # ==========================================
        # 4. Dimensions Scatter
        # ==========================================
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x='width', y='height', hue='class', alpha=0.6, s=60)
        plt.title('04 - Image Dimensions Correlation (Width vs Height)')
        plt.tight_layout()
        plt.savefig(split_out / '04_dimensions_scatter.png', dpi=200)
        plt.close()
        
        # ==========================================
        # 5. Aspect Ratio Density
        # ==========================================
        plt.figure(figsize=(10, 6))
        sns.kdeplot(
            data=df,
            x='aspect_ratio',
            hue='class',
            common_norm=False,
            fill=True,
            alpha=0.3,
            warn_singular=False,
        )
        plt.title('05 - Aspect Ratio Density Curves per Class')
        plt.tight_layout()
        plt.savefig(split_out / '05_aspect_ratio_kde.png', dpi=200)
        plt.close()
        
        # ==========================================
        # 6. File Size Boxplot
        # ==========================================
        plt.figure(figsize=(12, 6))
        _single_axis_palette_plot(
            sns.boxplot,
            data=df,
            x='class',
            y='file_size_kb',
            palette='Set2',
        )
        plt.title('06 - Image File Size Distribution per Class (KB)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(split_out / '06_filesize_boxplot.png', dpi=200)
        plt.close()
        
        # ==========================================
        # 7. Overall RGB Distribution
        # ==========================================
        plt.figure(figsize=(10, 6))
        sns.kdeplot(df['mean_r'], color='red', label='Red Channel', fill=True, alpha=0.3)
        sns.kdeplot(df['mean_g'], color='green', label='Green Channel', fill=True, alpha=0.3)
        sns.kdeplot(df['mean_b'], color='blue', label='Blue Channel', fill=True, alpha=0.3)
        plt.title('07 - Global RGB Channel Intensity Distributions')
        plt.legend()
        plt.tight_layout()
        plt.savefig(split_out / '07_rgb_distribution.png', dpi=200)
        plt.close()
        
        # ==========================================
        # 8. Class-wise Meaning RGB
        # ==========================================
        rgb_means = df.groupby('class')[['mean_r', 'mean_g', 'mean_b']].mean()
        rgb_means.plot(kind='bar', color=['red', 'green', 'blue'], figsize=(12, 6), alpha=0.7)
        plt.title('08 - Average RGB Values Separated per Class')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(split_out / '08_classwise_rgb_bar.png', dpi=200)
        plt.close()
        
        # ==========================================
        # 9. Brightness (Overall Mean Intensity)
        # ==========================================
        df['brightness'] = (df['mean_r'] + df['mean_g'] + df['mean_b']) / 3.0
        plt.figure(figsize=(12, 6))
        _single_axis_palette_plot(
            sns.violinplot,
            data=df,
            x='class',
            y='brightness',
            palette='pastel',
        )
        plt.title('09 - Brightness Violin Plot per Class')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(split_out / '09_brightness_violin.png', dpi=200)
        plt.close()
        
        # ==========================================
        # 10. Contrast (Std Dev)
        # ==========================================
        df['contrast'] = (df['std_r'] + df['std_g'] + df['std_b']) / 3.0
        plt.figure(figsize=(12, 6))
        _single_axis_palette_plot(
            sns.boxplot,
            data=df,
            x='class',
            y='contrast',
            palette='muted',
        )
        plt.title('10 - Image Contrast Variation (Std Dev) per Class')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(split_out / '10_contrast_boxplot.png', dpi=200)
        plt.close()
        
        # ==========================================
        # 11. Edge Density (Laplacian)
        # ==========================================
        plt.figure(figsize=(12, 6))
        # Add small epsilon to avoid log(0)
        df['laplacian_logged'] = np.log1p(df['laplacian'])
        sns.kdeplot(
            data=df,
            x='laplacian_logged',
            hue='class',
            common_norm=False,
            fill=True,
            alpha=0.3,
            warn_singular=False,
        )
        plt.title('11 - Edge Density / Sharpness (Log Laplacian Variance)')
        plt.tight_layout()
        plt.savefig(split_out / '11_sharpness_kde.png', dpi=200)
        plt.close()
        
        # ==========================================
        # 12. Average Image per Class (Ghost Image)
        # ==========================================
        fig, axes = plt.subplots(1, len(classes), figsize=(len(classes) * 3, 3))
        if len(classes) == 1: axes = [axes]
        for idx, cls in enumerate(classes):
            cls_df = df[df['class'] == cls]
            if not cls_df.empty:
                avg_img = np.mean(np.stack(cls_df['img_small'].values), axis=0).astype(np.uint8)
                axes[idx].imshow(avg_img)
                axes[idx].set_title(f"{cls}\n(Avg Pixels)")
            axes[idx].axis('off')
        plt.tight_layout()
        plt.savefig(split_out / '12_average_ghost_images.png', dpi=200)
        plt.close()
        
        # ==========================================
        # 13. Dominant Colors Extraction (K-Means)
        # ==========================================
        try:
            # Flatten all 64x64x3 images into just RGB pixels
            all_pixels = np.vstack(df['img_small'].apply(lambda x: x.reshape(-1, 3)).values)
            # Sample 20,000 pixels robustly
            if len(all_pixels) > 20000:
                indices = np.random.choice(len(all_pixels), 20000, replace=False)
                all_pixels = all_pixels[indices]
            
            kmeans = KMeans(n_clusters=8, random_state=42, n_init=10).fit(all_pixels)
            colors = kmeans.cluster_centers_.astype(int)
            
            # Sort colors by brightness for aesthetic display
            colors = sorted(colors, key=lambda x: sum(x))
            
            plt.figure(figsize=(10, 2))
            # Reshape for imshow
            plt.imshow([colors])
            plt.axis('off')
            plt.title(f'13 - Top 8 Dominant Extracted Colors in {split.upper()}')
            plt.tight_layout()
            plt.savefig(split_out / '13_dominant_colors_palette.png', dpi=200)
            plt.close()
        except Exception as e:
            print(f"Skipping dominant colors due to error: {e}")
        
    # ==========================================
    # 14. Cross-Split Comparison (Train vs Val Stacked Bar)
    # ==========================================
    if len(split_counts) == 2:
        df_counts = pd.DataFrame(split_counts).fillna(0)
        df_counts.plot(kind='bar', stacked=False, figsize=(12, 6), colormap='Set2')
        plt.title('14 - Train vs Val Cross-Split Class Distribution Comparison', fontsize=15)
        plt.ylabel('Number of Images')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / '14_train_vs_val_counts.png', dpi=200)
        plt.close()
        
    print(f"\nExploratory Data Analysis generated 14 distinct outputs seamlessly saved to {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Advanced Multi-Metric EDA on Dataset")
    parser.add_argument('--data-dir', type=str, default='Dataset_classes/1 Defined', help='Path to dataset directory')
    parser.add_argument('--out-dir', type=str, default='outputs/eda', help='Directory to output plots')
    args = parser.parse_args()
    
    create_eda_plots(args.data_dir, args.out_dir)

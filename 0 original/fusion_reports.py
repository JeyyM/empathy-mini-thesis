"""
Modular Fusion Emotion Report Generator
Generates separate detailed reports for fused multimodal emotion analysis
Color-coded according to emotion/dimension specifications (matching facial and voice)
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter
import seaborn as sns
import sys

class FusionReportGenerator:
    """Generate modular fusion emotion reports"""
    
    def __init__(self):
        # Color coding for emotions (MATCHING facial and voice)
        self.emotion_colors = {
            'angry': '#E74C3C',      # Red
            'surprise': '#FF8C00',   # Orange
            'happy': '#FFD700',      # Yellow
            'disgust': '#2ECC71',    # Green
            'sad': '#3498DB',        # Blue
            'fear': '#9370DB',       # Violet
            'neutral': '#95A5A6'     # Ash Gray
        }
        
        # Color coding for dimensions (MATCHING facial and voice)
        self.dimension_colors = {
            'intensity': '#E74C3C',  # Red
            'valence': '#3498DB',    # Blue
            'arousal': '#2ECC71',    # Green
            'stress': '#FF8C00'      # Orange
        }
        
        # Color coding for emotional states (MATCHING facial and voice)
        self.state_colors = {
            'Stressed': '#FF8C00',   # Orange
            'Excited': '#FFD700',    # Yellow
            'Tired': '#3498DB',      # Blue
            'Calm': '#2ECC71'        # Green
        }
        
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    
    def generate_emotion_report(self, df, base_filename):
        """
        Generate Page 1: Fused Emotion Analysis Report
        - Emotion over time of all 7 emotions
        - Highest overall emotion line graph
        - Summary statistics of all 7 emotions
        - Proportion pie graph and values
        """
        # Detect column prefix
        has_prefix = 'fused_angry' in df.columns
        prefix = 'fused_' if has_prefix else ''
        
        # Create figure
        fig = plt.figure(figsize=(20, 14))
        gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)
        
        fig.suptitle('🔮 MULTIMODAL FUSION EMOTION ANALYSIS - EMOTIONS (70% Facial + 30% Voice)', 
                    fontsize=22, fontweight='bold', y=0.98)
        
        # Get time data
        if 'time_seconds' in df.columns:
            time_data = df['time_seconds']
        else:
            time_data = np.arange(len(df))
        
        # ============ PLOT 1: All 7 Emotions Over Time ============
        ax1 = fig.add_subplot(gs[0, :])
        
        for emotion in self.emotions:
            col_name = f'{prefix}{emotion}'
            if col_name in df.columns:
                ax1.plot(time_data, df[col_name], 
                        label=emotion.capitalize(), 
                        color=self.emotion_colors[emotion],
                        linewidth=2.5, alpha=0.8)
        
        ax1.set_title('📊 All 7 Fused Emotions Over Time', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Time (seconds)', fontsize=12)
        ax1.set_ylabel('Emotion Probability (0-1)', fontsize=12)
        ax1.legend(loc='upper right', ncol=4, fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(-0.05, 1.05)
        
        # ============ PLOT 2: Dominant Emotion Only ============
        ax2 = fig.add_subplot(gs[1, 0])
        
        # Find dominant emotion at each time point
        emotion_cols = [f'{prefix}{e}' for e in self.emotions if f'{prefix}{e}' in df.columns]
        if emotion_cols:
            dominant_values = df[emotion_cols].max(axis=1)
            dominant_emotions = df[emotion_cols].idxmax(axis=1)
            
            # Plot with color coding
            for i in range(len(time_data)):
                emotion_name = dominant_emotions.iloc[i].replace(prefix, '')
                color = self.emotion_colors.get(emotion_name, '#95A5A6')
                if i < len(time_data) - 1:
                    ax2.plot(time_data.iloc[i:i+2], dominant_values.iloc[i:i+2], 
                            color=color, linewidth=3, alpha=0.9)
        
        ax2.set_title('⭐ Dominant Fused Emotion Over Time (Color-Coded)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Time (seconds)', fontsize=11)
        ax2.set_ylabel('Emotion Strength (0-1)', fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(-0.05, 1.05)
        
        # Add legend for colors
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=self.emotion_colors[e], label=e.capitalize()) 
                          for e in self.emotions]
        ax2.legend(handles=legend_elements, loc='upper right', ncol=2, fontsize=9)
        
        # ============ PLOT 3: Summary Statistics Table ============
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.axis('off')
        
        stats_data = []
        for emotion in self.emotions:
            col_name = f'{prefix}{emotion}'
            if col_name in df.columns:
                stats_data.append([
                    emotion.capitalize(),
                    f"{df[col_name].mean():.3f}",
                    f"{df[col_name].std():.3f}",
                    f"{df[col_name].median():.3f}",
                    f"{df[col_name].min():.3f}",
                    f"{df[col_name].max():.3f}"
                ])
        
        if stats_data:
            table = ax3.table(cellText=stats_data,
                            colLabels=['Emotion', 'Mean', 'Std Dev', 'Median', 'Min', 'Max'],
                            cellLoc='center',
                            loc='center',
                            bbox=[0, 0, 1, 1])
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2.5)
            
            # Color code rows by emotion
            for i, emotion in enumerate([e for e in self.emotions if f'{prefix}{e}' in df.columns], 1):
                table[(i, 0)].set_facecolor(self.emotion_colors[emotion])
                table[(i, 0)].set_text_props(color='white', weight='bold')
            
            # Style header
            for j in range(6):
                table[(0, j)].set_facecolor('#34495E')
                table[(0, j)].set_text_props(weight='bold', color='white')
        
        ax3.set_title('📈 Summary Statistics (All 7 Fused Emotions)', fontsize=14, fontweight='bold', pad=20)
        
        # ============ PLOT 4: Proportion Pie Chart ============
        ax4 = fig.add_subplot(gs[2, 0])
        
        # Calculate average proportions
        emotion_avgs = {}
        for emotion in self.emotions:
            col_name = f'{prefix}{emotion}'
            if col_name in df.columns:
                emotion_avgs[emotion] = df[col_name].mean()
        
        if emotion_avgs:
            colors = [self.emotion_colors[e] for e in emotion_avgs.keys()]
            labels = [f"{e.capitalize()}\n{v:.3f} ({v*100:.1f}%)" 
                     for e, v in emotion_avgs.items()]
            
            ax4.pie(emotion_avgs.values(), 
                   labels=labels,
                   colors=colors,
                   autopct='',
                   startangle=90,
                   textprops={'fontsize': 11, 'weight': 'bold'})
            
            ax4.set_title('🥧 Fused Emotion Distribution', 
                         fontsize=16, fontweight='bold', pad=20)
        
        # ============ PLOT 5: Distribution Table ============
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.axis('off')
        
        if emotion_avgs:
            # Create table data
            table_data = []
            total_avg = sum(emotion_avgs.values())
            
            for emotion in self.emotions:
                if emotion in emotion_avgs:
                    avg_value = emotion_avgs[emotion]
                    proportion = avg_value / total_avg if total_avg > 0 else 0
                    percentage = proportion * 100
                    table_data.append([
                        emotion.capitalize(),
                        f'{proportion:.4f}',
                        f'{percentage:.2f}%'
                    ])
            
            table = ax5.table(cellText=table_data,
                            colLabels=['Emotion', 'Proportion', 'Percentage'],
                            cellLoc='center',
                            loc='center',
                            bbox=[0, 0, 1, 1])
            table.auto_set_font_size(False)
            table.set_fontsize(11)
            table.scale(1, 2.5)
            
            # Color code emotion names
            for i, emotion in enumerate([e for e in self.emotions if e in emotion_avgs], 1):
                table[(i, 0)].set_facecolor(self.emotion_colors[emotion])
                table[(i, 0)].set_text_props(color='white', weight='bold')
            
            # Style header
            for j in range(3):
                table[(0, j)].set_facecolor('#34495E')
                table[(0, j)].set_text_props(weight='bold', color='white')
            
            ax5.set_title('📊 Fused Emotion Distribution Table', fontsize=14, fontweight='bold', pad=20)
            
            # Add dominant emotion text at bottom
            dominant_emotion = max(emotion_avgs.items(), key=lambda x: x[1])
            fig.text(0.5, 0.02, 
                    f"Most Dominant: {dominant_emotion[0].capitalize()} ({dominant_emotion[1]:.4f} - {dominant_emotion[1]*100:.2f}%)",
                    ha='center', fontsize=14, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', 
                             facecolor=self.emotion_colors[dominant_emotion[0]], 
                             edgecolor='black', linewidth=2))
        
        # Save
        save_path = f"{base_filename}_fusion_emotions.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Fusion Emotions report saved: {save_path}")
        plt.close()
    
    def generate_dimensions_report(self, df, base_filename):
        """
        Generate Page 2: Fused Dimensions Analysis Report
        - Intensity/Valence/Arousal/Stress timeline
        - Dimension summary statistics
        - Positivity/Negativity timeline
        - Stability metrics
        - Emotional volatility
        """
        # Detect column prefix
        has_prefix = 'fused_intensity' in df.columns
        prefix = 'fused_' if has_prefix else ''
        
        # Create figure
        fig = plt.figure(figsize=(20, 14))
        gs = GridSpec(4, 2, figure=fig, hspace=0.35, wspace=0.3)
        
        fig.suptitle('🔮 MULTIMODAL FUSION EMOTION ANALYSIS - DIMENSIONS (70% Facial + 30% Voice)', 
                    fontsize=22, fontweight='bold', y=0.98)
        
        # Get time data
        if 'time_seconds' in df.columns:
            time_data = df['time_seconds']
        else:
            time_data = np.arange(len(df))
        
        # ============ PLOT 1: Intensity, Valence, Arousal timeline (full width) ============
        ax1 = fig.add_subplot(gs[0, :])
        
        dimensions = [
            (f'{prefix}intensity', 'Intensity', self.dimension_colors['intensity']),
            (f'{prefix}valence', 'Valence', self.dimension_colors['valence']),
            (f'{prefix}arousal', 'Arousal', self.dimension_colors['arousal'])
        ]
        
        for col, label, color in dimensions:
            if col in df.columns:
                ax1.plot(time_data, df[col], label=label, color=color, linewidth=2.5, alpha=0.8)
        
        ax1.set_title('📊 Fused Emotion Dimensions Over Time', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Time (seconds)', fontsize=12)
        ax1.set_ylabel('Dimension Value', fontsize=12)
        ax1.legend(loc='upper right', fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
        
        # ============ PLOT 2: Dimension summary statistics (full width) ============
        ax2 = fig.add_subplot(gs[1, :])
        ax2.axis('off')
        
        stats_data = []
        all_dims = [f'{prefix}intensity', f'{prefix}valence', f'{prefix}arousal', f'{prefix}stress']
        dim_names = ['Intensity', 'Valence', 'Arousal', 'Stress']
        
        for col, name in zip(all_dims, dim_names):
            if col in df.columns:
                stats_data.append([
                    name,
                    f"{df[col].mean():.3f}",
                    f"{df[col].std():.3f}",
                    f"{df[col].median():.3f}",
                    f"{df[col].min():.3f}",
                    f"{df[col].max():.3f}"
                ])
        
        if stats_data:
            table = ax2.table(cellText=stats_data,
                            colLabels=['Dimension', 'Mean', 'Std Dev', 'Median', 'Min', 'Max'],
                            cellLoc='center',
                            loc='center',
                            bbox=[0.1, 0, 0.8, 1])
            table.auto_set_font_size(False)
            table.set_fontsize(11)
            table.scale(1, 2.5)
            
            # Color header
            for i in range(6):
                table[(0, i)].set_facecolor('#34495E')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            # Color dimension names
            for i, name in enumerate(dim_names[:len(stats_data)]):
                color = self.dimension_colors.get(name.lower(), '#808080')
                table[(i+1, 0)].set_facecolor(color)
                table[(i+1, 0)].set_text_props(weight='bold', color='white')
        
        ax2.set_title('📈 Fused Dimension Summary Statistics', fontsize=14, fontweight='bold', pad=20)
        
        # ============ PLOT 3: Positivity/Negativity timeline ============
        ax3 = fig.add_subplot(gs[2, 0])
        
        if f'{prefix}positivity' in df.columns and f'{prefix}negativity' in df.columns:
            ax3.plot(time_data, df[f'{prefix}positivity'], label='Positivity', 
                    color='#2ECC71', linewidth=2.5, alpha=0.8)
            ax3.plot(time_data, df[f'{prefix}negativity'], label='Negativity', 
                    color='#E74C3C', linewidth=2.5, alpha=0.8)
            ax3.fill_between(time_data, df[f'{prefix}positivity'], alpha=0.2, color='#2ECC71')
            ax3.fill_between(time_data, df[f'{prefix}negativity'], alpha=0.2, color='#E74C3C')
            
            ax3.set_title('⚖️ Fused Positivity vs Negativity', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Time (seconds)', fontsize=11)
            ax3.set_ylabel('Value', fontsize=11)
            ax3.legend(loc='upper right', fontsize=10)
            ax3.grid(True, alpha=0.3)
            ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # ============ PLOT 4: Stability metrics ============
        ax4 = fig.add_subplot(gs[2, 1])
        ax4.axis('off')
        
        stability_data = []
        
        for col, name in zip(all_dims, dim_names):
            if col in df.columns:
                # Calculate stability (inverse of variance)
                variance = df[col].var()
                stability = 1 / (1 + variance) if variance > 0 else 1.0
                
                # Calculate consistency (inverse of std dev)
                std = df[col].std()
                consistency = 1 / (1 + std) if std > 0 else 1.0
                
                stability_data.append([
                    name,
                    f"{stability:.3f}",
                    f"{consistency:.3f}",
                    f"{variance:.3f}"
                ])
        
        if stability_data:
            table = ax4.table(cellText=stability_data,
                            colLabels=['Dimension', 'Stability', 'Consistency', 'Variance'],
                            cellLoc='center',
                            loc='center',
                            bbox=[0, 0, 1, 1])
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2.5)
            
            # Color header
            for i in range(4):
                table[(0, i)].set_facecolor('#34495E')
                table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax4.set_title('📊 Fused Emotion Stability Metrics', fontsize=14, fontweight='bold', pad=20)
        
        # ============ PLOT 5: Emotional volatility over time (full width) ============
        ax5 = fig.add_subplot(gs[3, :])
        
        if f'{prefix}intensity' in df.columns:
            # Calculate rolling volatility (standard deviation over window)
            window = min(20, len(df) // 10)
            if window > 1:
                volatility = df[f'{prefix}intensity'].rolling(window=window, min_periods=1).std()
                ax5.plot(time_data, volatility, color='#E91E63', linewidth=2.5, alpha=0.8)
                ax5.fill_between(time_data, volatility, alpha=0.3, color='#E91E63')
                
                ax5.set_title(f'📉 Fused Emotional Volatility Over Time (Rolling Std Dev, Window={window})', 
                            fontsize=14, fontweight='bold')
                ax5.set_xlabel('Time (seconds)', fontsize=11)
                ax5.set_ylabel('Volatility', fontsize=11)
                ax5.grid(True, alpha=0.3)
        
        # Save
        save_path = f"{base_filename}_fusion_dimensions.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Fusion Dimensions report saved: {save_path}")
        plt.close()
    
    def generate_states_report(self, df, base_filename):
        """
        Generate Page 3: Fused Emotional States Report (4x4 grid)
        - Circular journey path (2x2)
        - State summary statistics table
        - Dimension statistics table
        - Average state values bar chart
        - State over time timeline
        - State proportion pie chart
        - State distribution table
        """
        # Detect column prefix
        has_prefix = 'fused_quadrant' in df.columns
        prefix = 'fused_' if has_prefix else ''
        
        # Create figure
        fig = plt.figure(figsize=(20, 20))
        gs = GridSpec(4, 4, figure=fig, hspace=0.4, wspace=0.4)
        
        fig.suptitle('🔮 MULTIMODAL FUSION EMOTION ANALYSIS - STATES (70% Facial + 30% Voice)', 
                    fontsize=22, fontweight='bold', y=0.98)
        
        # Get time data
        if 'time_seconds' in df.columns:
            time_data = df['time_seconds']
        else:
            time_data = np.arange(len(df))
        
        # ============ Row 0-1, Col 0-1: Circular journey path (2x2) ============
        ax_journey = plt.subplot(gs[0:2, 0:2])
        
        arousal_col = f'{prefix}arousal'
        valence_col = f'{prefix}valence'
        
        if arousal_col in df.columns and valence_col in df.columns:
            arousal = df[arousal_col].values
            valence = df[valence_col].values
            
            # Create circular boundary
            theta = np.linspace(0, 2*np.pi, 100)
            circle_x = np.cos(theta)
            circle_y = np.sin(theta)
            ax_journey.plot(circle_x, circle_y, 'k-', linewidth=2, alpha=0.5)
            
            # Create heatmap background
            grid_size = 100
            x_grid = np.linspace(-1, 1, grid_size)
            y_grid = np.linspace(-1, 1, grid_size)
            X, Y = np.meshgrid(x_grid, y_grid)
            
            # Calculate density heatmap
            heatmap = np.zeros((grid_size, grid_size))
            for v, a in zip(valence, arousal):
                # Normalize to -1 to 1 range
                v_norm = np.clip(v, -1, 1)
                a_norm = np.clip(a, -1, 1)
                
                # Find grid position
                x_idx = int((v_norm + 1) * (grid_size - 1) / 2)
                y_idx = int((a_norm + 1) * (grid_size - 1) / 2)
                
                x_idx = np.clip(x_idx, 0, grid_size - 1)
                y_idx = np.clip(y_idx, 0, grid_size - 1)
                
                heatmap[y_idx, x_idx] += 1
            
            # Apply Gaussian smoothing
            heatmap_smooth = gaussian_filter(heatmap, sigma=3)
            
            # Mask outside circle
            mask = (X**2 + Y**2) > 1
            heatmap_smooth_masked = np.ma.array(heatmap_smooth, mask=mask)
            
            # Plot heatmap
            im = ax_journey.contourf(X, Y, heatmap_smooth_masked, levels=20, 
                                    cmap='plasma', alpha=0.6)
            
            # Plot journey path (traveling line)
            valence_norm = np.clip(valence, -1, 1)
            arousal_norm = np.clip(arousal, -1, 1)
            ax_journey.plot(valence_norm, arousal_norm, 'w-', linewidth=2.5, 
                           alpha=0.9, label='Emotional Journey')
            
            # Mark start and end
            ax_journey.scatter(valence_norm[0], arousal_norm[0], 
                              color='lime', s=300, marker='o', 
                              edgecolors='white', linewidths=2.5, 
                              label='Start', zorder=5)
            ax_journey.scatter(valence_norm[-1], arousal_norm[-1], 
                              color='red', s=300, marker='s', 
                              edgecolors='white', linewidths=2.5, 
                              label='End', zorder=5)
            
            # Add quadrant labels with colors
            ax_journey.text(0.5, 0.5, 'Excited', fontsize=16, fontweight='bold',
                           ha='center', va='center', color=self.state_colors['Excited'],
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
            ax_journey.text(-0.5, 0.5, 'Stressed', fontsize=16, fontweight='bold',
                           ha='center', va='center', color=self.state_colors['Stressed'],
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
            ax_journey.text(-0.5, -0.5, 'Tired', fontsize=16, fontweight='bold',
                           ha='center', va='center', color=self.state_colors['Tired'],
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
            ax_journey.text(0.5, -0.5, 'Calm', fontsize=16, fontweight='bold',
                           ha='center', va='center', color=self.state_colors['Calm'],
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
            
            # Add axis lines
            ax_journey.axhline(y=0, color='black', linestyle='--', alpha=0.3, linewidth=1.5)
            ax_journey.axvline(x=0, color='black', linestyle='--', alpha=0.3, linewidth=1.5)
            
            ax_journey.set_xlabel('Valence (Negative ← → Positive)', fontsize=13, fontweight='bold')
            ax_journey.set_ylabel('Arousal (Low ← → High)', fontsize=13, fontweight='bold')
            ax_journey.set_title('🗺️ Fused Emotional Journey Path', fontsize=16, fontweight='bold', pad=15)
            ax_journey.set_xlim(-1.1, 1.1)
            ax_journey.set_ylim(-1.1, 1.1)
            ax_journey.set_aspect('equal')
            ax_journey.legend(loc='upper left', fontsize=11, framealpha=0.9)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax_journey, fraction=0.046, pad=0.04)
            cbar.set_label('Movement Intensity', fontsize=11, fontweight='bold')
        
        # ============ Row 0, Col 2: State summary statistics table ============
        ax_state_stats = plt.subplot(gs[0, 2:])
        ax_state_stats.axis('off')
        
        quadrant_col = f'{prefix}quadrant'
        if quadrant_col in df.columns:
            state_counts = df[quadrant_col].value_counts()
            stats_data = []
            
            all_states = ['Stressed', 'Excited', 'Tired', 'Calm']
            for state in all_states:
                count = state_counts.get(state, 0)
                percentage = (count / len(df) * 100) if len(df) > 0 else 0
                stats_data.append([state, f"{count}", f"{percentage:.1f}%"])
            
            table = ax_state_stats.table(cellText=stats_data,
                                        colLabels=['State', 'Count', 'Percentage'],
                                        cellLoc='center',
                                        loc='center',
                                        bbox=[0, 0, 1, 1])
            table.auto_set_font_size(False)
            table.set_fontsize(11)
            table.scale(1, 2.5)
            
            # Color header
            for i in range(3):
                table[(0, i)].set_facecolor('#34495E')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            # Color state names
            for i, state in enumerate(all_states):
                color = self.state_colors.get(state, '#808080')
                table[(i+1, 0)].set_facecolor(color)
                table[(i+1, 0)].set_text_props(weight='bold', color='white')
            
            ax_state_stats.set_title('📊 Fused State Summary Statistics', fontsize=14, fontweight='bold', pad=20)
        
        # ============ Row 1, Col 2: Dimension statistics table ============
        ax_dim_stats = plt.subplot(gs[1, 2:])
        ax_dim_stats.axis('off')
        
        dim_data = []
        dimensions = [
            (f'{prefix}arousal', 'Arousal', self.dimension_colors['arousal']),
            (f'{prefix}valence', 'Valence', self.dimension_colors['valence']),
            (f'{prefix}intensity', 'Intensity', self.dimension_colors['intensity']),
            (f'{prefix}stress', 'Stress', self.dimension_colors['stress'])
        ]
        
        for col, name, color in dimensions:
            if col in df.columns:
                dim_data.append([
                    name,
                    f"{df[col].mean():.3f}",
                    f"{df[col].std():.3f}"
                ])
        
        if dim_data:
            table = ax_dim_stats.table(cellText=dim_data,
                                       colLabels=['Dimension', 'Mean', 'Std Dev'],
                                       cellLoc='center',
                                       loc='center',
                                       bbox=[0, 0, 1, 1])
            table.auto_set_font_size(False)
            table.set_fontsize(11)
            table.scale(1, 2.5)
            
            # Color header
            for i in range(3):
                table[(0, i)].set_facecolor('#34495E')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            # Color dimension names
            for i, (col, name, color) in enumerate(dimensions[:len(dim_data)]):
                table[(i+1, 0)].set_facecolor(color)
                table[(i+1, 0)].set_text_props(weight='bold', color='white')
            
            ax_dim_stats.set_title('📈 Fused Dimension Statistics', fontsize=14, fontweight='bold', pad=20)
        
        # ============ Row 2, Col 0: Average state values bar chart ============
        ax_bar = plt.subplot(gs[2, 0])
        
        if quadrant_col in df.columns:
            state_counts = df[quadrant_col].value_counts()
            all_states = ['Stressed', 'Excited', 'Tired', 'Calm']
            counts = [state_counts.get(state, 0) for state in all_states]
            colors = [self.state_colors[state] for state in all_states]
            
            bars = ax_bar.bar(all_states, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax_bar.text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height)}',
                           ha='center', va='bottom', fontweight='bold', fontsize=11)
            
            ax_bar.set_title('📊 Fused State Distribution', fontsize=14, fontweight='bold')
            ax_bar.set_xlabel('Emotional State', fontsize=11)
            ax_bar.set_ylabel('Sample Count', fontsize=11)
            ax_bar.grid(True, alpha=0.3, axis='y')
            ax_bar.tick_params(axis='x', labelsize=10)
        
        # ============ Row 2, Col 1-3: State over time timeline ============
        ax_timeline = plt.subplot(gs[2, 1:])
        
        if quadrant_col in df.columns:
            # Map states to numeric values for plotting
            state_map = {'Stressed': 3, 'Excited': 2, 'Tired': 1, 'Calm': 0}
            state_values = df[quadrant_col].map(state_map)
            colors = [self.state_colors.get(s, '#808080') for s in df[quadrant_col]]
            
            ax_timeline.scatter(time_data, state_values, c=colors, s=50, alpha=0.8, 
                               edgecolors='black', linewidth=0.5)
            
            ax_timeline.set_title('⏱️ Fused Emotional State Over Time', fontsize=14, fontweight='bold')
            ax_timeline.set_xlabel('Time (seconds)', fontsize=11)
            ax_timeline.set_ylabel('Emotional State', fontsize=11)
            ax_timeline.set_yticks([0, 1, 2, 3])
            ax_timeline.set_yticklabels(['Calm', 'Tired', 'Excited', 'Stressed'])
            ax_timeline.grid(True, alpha=0.3, axis='x')
        
        # ============ Row 3, Col 0: State proportion pie chart ============
        ax_pie = plt.subplot(gs[3, 0])
        
        if quadrant_col in df.columns:
            state_counts = df[quadrant_col].value_counts()
            colors = [self.state_colors.get(state, '#808080') for state in state_counts.index]
            
            wedges, texts, autotexts = ax_pie.pie(state_counts.values,
                                                   labels=state_counts.index,
                                                   colors=colors,
                                                   autopct='%1.1f%%',
                                                   startangle=90,
                                                   textprops={'fontsize': 11, 'weight': 'bold'})
            
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(11)
            
            ax_pie.set_title('🥧 Fused State Proportion', fontsize=14, fontweight='bold')
        
        # ============ Row 3, Col 1-3: State distribution table ============
        ax_dist_table = plt.subplot(gs[3, 1:])
        ax_dist_table.axis('off')
        
        if quadrant_col in df.columns:
            state_counts = df[quadrant_col].value_counts()
            dist_data = []
            
            all_states = ['Stressed', 'Excited', 'Tired', 'Calm']
            for state in all_states:
                count = state_counts.get(state, 0)
                proportion = (count / len(df)) if len(df) > 0 else 0
                percentage = proportion * 100
                dist_data.append([
                    state,
                    f"{count}",
                    f"{proportion:.4f}",
                    f"{percentage:.2f}%"
                ])
            
            table = ax_dist_table.table(cellText=dist_data,
                                        colLabels=['State', 'Count', 'Proportion', 'Percentage'],
                                        cellLoc='center',
                                        loc='center',
                                        bbox=[0.1, 0, 0.8, 1])
            table.auto_set_font_size(False)
            table.set_fontsize(11)
            table.scale(1, 2.5)
            
            # Color header
            for i in range(4):
                table[(0, i)].set_facecolor('#34495E')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            # Color state names
            for i, state in enumerate(all_states):
                color = self.state_colors.get(state, '#808080')
                table[(i+1, 0)].set_facecolor(color)
                table[(i+1, 0)].set_text_props(weight='bold', color='white')
            
            ax_dist_table.set_title('📊 Fused State Distribution Details', 
                                   fontsize=14, fontweight='bold', pad=20)
        
        # Save
        save_path = f"{base_filename}_fusion_states.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Fusion States report saved: {save_path}")
        plt.close()


def generate_all_reports(csv_file):
    """
    Generate all three fusion reports
    
    Args:
        csv_file: Path to fusion CSV file
    """
    print("\n" + "="*80)
    print("📊 GENERATING FUSED EMOTION REPORTS")
    print("="*80)
    print(f"📂 Input file: {csv_file}\n")
    
    # Load data
    df = pd.read_csv(csv_file)
    print(f"✓ Loaded {len(df)} samples\n")
    
    # Generate base filename
    base = csv_file.replace('_fusion.csv', '').replace('.csv', '')
    
    # Create generator instance
    generator = FusionReportGenerator()
    
    # Generate reports
    generator.generate_emotion_report(df, base)
    generator.generate_dimensions_report(df, base)
    generator.generate_states_report(df, base)
    
    print("\n" + "="*80)
    print("✅ ALL FUSION REPORTS GENERATED SUCCESSFULLY!")
    print("="*80)
    print(f"\n📊 Generated files:")
    print(f"  1. {base}_fusion_emotions.png")
    print(f"  2. {base}_fusion_dimensions.png")
    print(f"  3. {base}_fusion_states.png")
    print("\n")


def main():
    """Command-line interface"""
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        print("Usage: python fusion_reports.py <fusion_csv>")
        print("Example: python fusion_reports.py angry_fusion.csv")
        return
    
    generate_all_reports(csv_file)


if __name__ == "__main__":
    main()

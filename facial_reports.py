"""
Modular Facial Emotion Report Generator
Generates separate detailed reports for different aspects of facial emotion analysis
Color-coded according to emotion/dimension specifications
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
import seaborn as sns

class FacialReportGenerator:
    """Generate modular facial emotion reports"""
    
    def __init__(self):
        # Color coding for emotions
        self.emotion_colors = {
            'angry': '#E74C3C',      # Red
            'surprise': '#FF8C00',   # Orange
            'happy': '#FFD700',      # Yellow
            'disgust': '#2ECC71',    # Green
            'sad': '#3498DB',        # Blue
            'fear': '#9370DB',       # Violet
            'neutral': '#95A5A6'     # Ash Gray
        }
        
        # Color coding for dimensions
        self.dimension_colors = {
            'intensity': '#E74C3C',  # Red
            'valence': '#3498DB',    # Blue
            'arousal': '#2ECC71'     # Green
        }
        
        # Color coding for emotional states
        self.state_colors = {
            'Stressed': '#FF8C00',   # Orange
            'Excited': '#FFD700',    # Yellow
            'Tired': '#3498DB',      # Blue
            'Calm': '#2ECC71'        # Green
        }
        
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    
    def generate_emotion_report(self, df, base_filename):
        """
        Generate Page 1: Emotion Analysis Report
        - Emotion over time of all 7 emotions
        - Highest overall emotion line graph
        - Summary statistics of all 7 emotions
        - Proportion pie graph and values
        """
        # Detect column prefix
        has_prefix = 'facial_angry' in df.columns
        prefix = 'facial_' if has_prefix else ''
        
        # Create figure
        fig = plt.figure(figsize=(20, 14))
        gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)
        
        fig.suptitle('🎭 FACIAL EMOTION ANALYSIS - EMOTIONS', 
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
        
        ax1.set_title('📊 All 7 Emotions Over Time', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Time (seconds)', fontsize=12)
        ax1.set_ylabel('Emotion Probability (0-1)', fontsize=12)
        ax1.legend(loc='upper right', ncol=4, fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(-0.05, 1.05)
        
        # ============ PLOT 2: Dominant Emotion Only ============
        ax2 = fig.add_subplot(gs[1, 0])
        
        # Find dominant emotion at each time point
        emotion_cols = [f'{prefix}{e}' for e in self.emotions if f'{prefix}{e}' in df.columns]
        dominant_values = df[emotion_cols].max(axis=1)
        dominant_emotions = df[emotion_cols].idxmax(axis=1)
        
        # Plot with color coding
        for i in range(len(time_data)):
            emotion_name = dominant_emotions.iloc[i].replace(prefix, '')
            color = self.emotion_colors.get(emotion_name, '#95A5A6')
            ax2.plot(time_data.iloc[i:i+2] if i < len(time_data)-1 else [time_data.iloc[i]], 
                    dominant_values.iloc[i:i+2] if i < len(dominant_values)-1 else [dominant_values.iloc[i]], 
                    color=color, linewidth=3, alpha=0.9)
        
        ax2.set_title('⭐ Dominant Emotion Over Time (Color-Coded)', fontsize=14, fontweight='bold')
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
        
        ax3.set_title('📈 Summary Statistics (All 7 Emotions)', fontsize=14, fontweight='bold', pad=20)
        
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
            
            wedges, texts, autotexts = ax4.pie(emotion_avgs.values(), 
                                               labels=labels,
                                               colors=colors,
                                               autopct='',
                                               startangle=90,
                                               textprops={'fontsize': 11, 'weight': 'bold'})
            
            # Make percentages white and bold
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontsize(12)
                autotext.set_weight('bold')
            
            ax4.set_title('🥧 Emotion Distribution - Pie Chart', 
                         fontsize=16, fontweight='bold', pad=20)
        
        # ============ PLOT 5: Distribution Table ============
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.axis('off')
        
        if emotion_avgs:
            # Sort by value descending
            sorted_emotions = sorted(emotion_avgs.items(), key=lambda x: x[1], reverse=True)
            
            table_data = []
            for emotion, value in sorted_emotions:
                table_data.append([
                    emotion.capitalize(),
                    f"{value:.4f}",
                    f"{value*100:.2f}%"
                ])
            
            table = ax5.table(cellText=table_data,
                            colLabels=['Emotion', 'Value', 'Percentage'],
                            cellLoc='center',
                            loc='center',
                            bbox=[0, 0, 1, 1])
            table.auto_set_font_size(False)
            table.set_fontsize(11)
            table.scale(1, 2.8)
            
            # Color code emotion names
            for i, (emotion, _) in enumerate(sorted_emotions, 1):
                table[(i, 0)].set_facecolor(self.emotion_colors[emotion])
                table[(i, 0)].set_text_props(color='white', weight='bold')
            
            # Style header
            for j in range(3):
                table[(0, j)].set_facecolor('#34495E')
                table[(0, j)].set_text_props(weight='bold', color='white')
            
            ax5.set_title('📊 Emotion Distribution - Table', fontsize=16, fontweight='bold', pad=20)
            
            # Add dominant emotion text
            dominant_emotion = sorted_emotions[0]
            fig.text(0.5, 0.02, 
                    f"Most Dominant: {dominant_emotion[0].capitalize()} ({dominant_emotion[1]:.3f} - {dominant_emotion[1]*100:.1f}%)",
                    ha='center', fontsize=14, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', 
                             facecolor=self.emotion_colors[dominant_emotion[0]], 
                             edgecolor='black', linewidth=2))
        
        # Save
        save_path = f"{base_filename}_facial_emotions.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Facial Emotions report saved: {save_path}")
        plt.close()
    
    def generate_dimensions_report(self, df, base_filename):
        """
        Generate Page 2: Dimensions Analysis Report
        - Intensity, valence, arousal over time
        - Intensity, valence, arousal summary statistics
        - Positivity and negativity over time
        - Emotion stability metrics
        """
        # Detect column prefix
        has_prefix = 'facial_intensity' in df.columns
        prefix = 'facial_' if has_prefix else ''
        
        # Create figure
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(4, 2, figure=fig, hspace=0.4, wspace=0.3)
        
        fig.suptitle('🎭 FACIAL EMOTION ANALYSIS - DIMENSIONS', 
                    fontsize=22, fontweight='bold', y=0.98)
        
        # Get time data
        if 'time_seconds' in df.columns:
            time_data = df['time_seconds']
        else:
            time_data = np.arange(len(df))
        
        # ============ PLOT 1: Intensity, Valence, Arousal Over Time ============
        ax1 = fig.add_subplot(gs[0, :])
        
        dimensions = ['intensity', 'valence', 'arousal']
        for dim in dimensions:
            col_name = f'{prefix}{dim}'
            if col_name in df.columns:
                ax1.plot(time_data, df[col_name], 
                        label=dim.capitalize(),
                        color=self.dimension_colors[dim],
                        linewidth=3, alpha=0.8)
        
        ax1.set_title('📊 Emotional Dimensions Over Time', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Time (seconds)', fontsize=12)
        ax1.set_ylabel('Dimension Value', fontsize=12)
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # ============ PLOT 2: Dimensions Summary Statistics Table ============
        ax2 = fig.add_subplot(gs[1, :])
        ax2.axis('off')
        
        stats_data = []
        for dim in dimensions:
            col_name = f'{prefix}{dim}'
            if col_name in df.columns:
                stats_data.append([
                    dim.capitalize(),
                    f"{df[col_name].mean():.4f}",
                    f"{df[col_name].std():.4f}",
                    f"{df[col_name].median():.4f}",
                    f"{df[col_name].min():.4f}",
                    f"{df[col_name].max():.4f}",
                    f"{df[col_name].max() - df[col_name].min():.4f}"
                ])
        
        if stats_data:
            table = ax2.table(cellText=stats_data,
                            colLabels=['Dimension', 'Mean', 'Std Dev', 'Median', 'Min', 'Max', 'Range'],
                            cellLoc='center',
                            loc='center',
                            bbox=[0, 0, 1, 1])
            table.auto_set_font_size(False)
            table.set_fontsize(11)
            table.scale(1, 3.5)
            
            # Color code dimension names
            for i, dim in enumerate([d for d in dimensions if f'{prefix}{d}' in df.columns], 1):
                table[(i, 0)].set_facecolor(self.dimension_colors[dim])
                table[(i, 0)].set_text_props(color='white', weight='bold')
            
            # Style header
            for j in range(7):
                table[(0, j)].set_facecolor('#34495E')
                table[(0, j)].set_text_props(weight='bold', color='white')
        
        ax2.set_title('📈 Intensity, Valence, and Arousal - Summary Statistics', 
                     fontsize=16, fontweight='bold', pad=20)
        
        # ============ PLOT 3: Positivity & Negativity Over Time ============
        ax3 = fig.add_subplot(gs[2, 0])
        
        if f'{prefix}positivity' in df.columns and f'{prefix}negativity' in df.columns:
            ax3.plot(time_data, df[f'{prefix}positivity'], 
                    label='Positivity', color='#2ECC71', linewidth=3, alpha=0.8)
            ax3.plot(time_data, df[f'{prefix}negativity'], 
                    label='Negativity', color='#E74C3C', linewidth=3, alpha=0.8)
            ax3.fill_between(time_data, df[f'{prefix}positivity'], 0, 
                           color='#2ECC71', alpha=0.2)
            ax3.fill_between(time_data, df[f'{prefix}negativity'], 0, 
                           color='#E74C3C', alpha=0.2)
            
            ax3.set_title('⚖️ Positivity vs Negativity Over Time', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Time (seconds)', fontsize=11)
            ax3.set_ylabel('Value', fontsize=11)
            ax3.legend(fontsize=11)
            ax3.grid(True, alpha=0.3)
            ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # ============ PLOT 4: Emotion Stability Metrics ============
        ax4 = fig.add_subplot(gs[2, 1])
        ax4.axis('off')
        
        # Calculate stability metrics
        stability_data = []
        
        # Count emotion changes (when dominant emotion switches)
        emotion_cols = [f'{prefix}{e}' for e in self.emotions if f'{prefix}{e}' in df.columns]
        if emotion_cols:
            dominant_emotions = df[emotion_cols].idxmax(axis=1)
            emotion_changes = (dominant_emotions != dominant_emotions.shift()).sum() - 1
            stability_data.append(['Emotion Changes', f'{emotion_changes}'])
        
        # Standard deviation over time for each emotion
        for emotion in self.emotions:
            col_name = f'{prefix}{emotion}'
            if col_name in df.columns:
                stability_data.append([f'{emotion.capitalize()} Std Dev', f'{df[col_name].std():.4f}'])
        
        # Overall emotional volatility (mean of all std devs)
        if emotion_cols:
            volatility = df[emotion_cols].std().mean()
            stability_data.append(['Overall Volatility', f'{volatility:.4f}'])
        
        if stability_data:
            table = ax4.table(cellText=stability_data,
                            colLabels=['Stability Metric', 'Value'],
                            cellLoc='left',
                            loc='center',
                            bbox=[0, 0, 1, 1])
            table.auto_set_font_size(False)
            table.set_fontsize(11)
            table.scale(1, 2.5)
            
            # Style header
            for j in range(2):
                table[(0, j)].set_facecolor('#34495E')
                table[(0, j)].set_text_props(weight='bold', color='white')
            
            # Alternate row colors
            for i in range(1, len(stability_data) + 1):
                for j in range(2):
                    if i % 2 == 0:
                        table[(i, j)].set_facecolor('#ECF0F1')
        
        ax4.set_title('📊 Emotion Stability Metrics', fontsize=14, fontweight='bold', pad=20)
        
        # ============ PLOT 5: Standard Deviation Over Time (Rolling) ============
        ax5 = fig.add_subplot(gs[3, :])
        
        if emotion_cols:
            # Calculate rolling standard deviation
            window = min(30, len(df) // 10)  # Adaptive window size
            rolling_std = df[emotion_cols].rolling(window=window, center=True).std().mean(axis=1)
            
            ax5.plot(time_data, rolling_std, color='#E74C3C', linewidth=2.5, alpha=0.8)
            ax5.fill_between(time_data, rolling_std, 0, color='#E74C3C', alpha=0.2)
            ax5.set_title(f'📈 Emotional Volatility Over Time (Rolling Std Dev, window={window})', 
                         fontsize=14, fontweight='bold')
            ax5.set_xlabel('Time (seconds)', fontsize=12)
            ax5.set_ylabel('Standard Deviation', fontsize=12)
            ax5.grid(True, alpha=0.3)
        
        # Save
        save_path = f"{base_filename}_facial_dimensions.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Facial Dimensions report saved: {save_path}")
        plt.close()
    
    def generate_states_report(self, df, base_filename):
        """
        Generate Page 3: Emotional States Report
        - Emotional state movement heatmap (circular)
        - Summary statistics table
        - Average emotional state values
        - Current emotional state over time
        - Emotional state proportion with distribution table
        """
        # Detect column prefix
        has_prefix = 'facial_arousal' in df.columns
        prefix = 'facial_' if has_prefix else ''
        
        # Create figure
        fig = plt.figure(figsize=(24, 18))
        gs = GridSpec(4, 4, figure=fig, hspace=0.4, wspace=0.35)
        
        fig.suptitle('🎭 FACIAL EMOTION ANALYSIS - EMOTIONAL STATES', 
                    fontsize=22, fontweight='bold', y=0.98)
        
        # Get time data
        if 'time_seconds' in df.columns:
            time_data = df['time_seconds']
        else:
            time_data = np.arange(len(df))
        
        # Check if quadrant data exists, if not create it
        quadrant_col = f'{prefix}quadrant'
        if quadrant_col not in df.columns:
            # Create quadrants from arousal and valence
            if f'{prefix}arousal' in df.columns and f'{prefix}valence' in df.columns:
                df[quadrant_col] = df.apply(lambda row: 
                    'Excited' if row[f'{prefix}arousal'] > 0 and row[f'{prefix}valence'] > 0
                    else 'Stressed' if row[f'{prefix}arousal'] > 0 and row[f'{prefix}valence'] < 0
                    else 'Calm' if row[f'{prefix}arousal'] < 0 and row[f'{prefix}valence'] > 0
                    else 'Tired', axis=1)
        
        # ============ PLOT 1: Emotional Journey Path (Circular) ============
        ax1 = fig.add_subplot(gs[0:2, 0:2])  # Takes 2 rows, 2 columns
        
        if f'{prefix}arousal' in df.columns and f'{prefix}valence' in df.columns:
            arousal = df[f'{prefix}arousal'].values
            valence = df[f'{prefix}valence'].values
            
            # Create circular grid for background heatmap
            grid_size = 50
            x = np.linspace(-1, 1, grid_size)
            y = np.linspace(-1, 1, grid_size)
            X, Y = np.meshgrid(x, y)
            
            # Create mask for circular boundary
            circle_mask = X**2 + Y**2 <= 1
            
            # Create smooth heatmap based on trajectory
            gaussian_grid = np.zeros((grid_size, grid_size))
            
            for i, (val, ar) in enumerate(zip(valence, arousal)):
                # Convert to grid coordinates
                x_idx = int((val + 1) * grid_size / 2)
                y_idx = int((ar + 1) * grid_size / 2)
                
                # Ensure indices are within bounds
                x_idx = max(0, min(grid_size - 1, x_idx))
                y_idx = max(0, min(grid_size - 1, y_idx))
                
                # Add Gaussian blob around the point
                sigma = 3  # Spread of the Gaussian
                for dx in range(-sigma*2, sigma*2 + 1):
                    for dy in range(-sigma*2, sigma*2 + 1):
                        nx, ny = x_idx + dx, y_idx + dy
                        if 0 <= nx < grid_size and 0 <= ny < grid_size:
                            distance = np.sqrt(dx**2 + dy**2)
                            weight = np.exp(-distance**2 / (2 * sigma**2))
                            gaussian_grid[ny, nx] += weight
            
            # Apply circular mask
            gaussian_masked = np.where(circle_mask, gaussian_grid, np.nan)
            
            im = ax1.imshow(gaussian_masked, extent=[-1, 1, -1, 1], origin='lower', cmap='plasma', alpha=0.8)
            
            # Draw movement path
            ax1.plot(valence, arousal, 'white', linewidth=2.5, alpha=0.9, label='Emotion Path')
            ax1.scatter(valence[0], arousal[0], color='lime', s=150, marker='o', 
                       label='Start', edgecolor='black', linewidth=2, zorder=5)
            ax1.scatter(valence[-1], arousal[-1], color='red', s=150, marker='s', 
                       label='End', edgecolor='black', linewidth=2, zorder=5)
            
            # Draw circle boundary
            circle = plt.Circle((0, 0), 1, fill=False, color='white', linewidth=3)
            ax1.add_patch(circle)
            
            # Draw quadrant lines
            ax1.axhline(y=0, color='white', linestyle='-', alpha=0.7, linewidth=2)
            ax1.axvline(x=0, color='white', linestyle='-', alpha=0.7, linewidth=2)
            
            # Add quadrant labels with color-coded backgrounds
            ax1.text(0.6, 0.6, 'EXCITED', fontsize=14, ha='center', va='center', 
                    fontweight='bold', color='white',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor=self.state_colors['Excited'], 
                             edgecolor='black', linewidth=2, alpha=0.9))
            ax1.text(-0.6, 0.6, 'STRESSED', fontsize=14, ha='center', va='center',
                    fontweight='bold', color='white',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor=self.state_colors['Stressed'], 
                             edgecolor='black', linewidth=2, alpha=0.9))
            ax1.text(0.6, -0.6, 'CALM', fontsize=14, ha='center', va='center',
                    fontweight='bold', color='white',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor=self.state_colors['Calm'], 
                             edgecolor='black', linewidth=2, alpha=0.9))
            ax1.text(-0.6, -0.6, 'TIRED', fontsize=14, ha='center', va='center',
                    fontweight='bold', color='white',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor=self.state_colors['Tired'], 
                             edgecolor='black', linewidth=2, alpha=0.9))
            
            ax1.set_xlabel('Valence (Negative ← → Positive)', fontsize=13, fontweight='bold')
            ax1.set_ylabel('Arousal (Low ← → High)', fontsize=13, fontweight='bold')
            ax1.set_title('🛤️ Emotional Journey Path', fontsize=16, fontweight='bold', pad=15)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax1, shrink=0.7)
            cbar.set_label('Movement Intensity', fontsize=11)
            
            ax1.legend(loc='upper left', fontsize=10)
            ax1.set_xlim(-1.1, 1.1)
            ax1.set_ylim(-1.1, 1.1)
            ax1.set_aspect('equal')
        
        # ============ PLOT 2: Summary Statistics Table ============
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.axis('off')
        
        stats_data = []
        if quadrant_col in df.columns:
            state_counts = df[quadrant_col].value_counts()
            total_samples = len(df)
            
            # Include all states, even with 0 count
            for state in ['Stressed', 'Excited', 'Tired', 'Calm']:
                count = state_counts.get(state, 0)
                proportion = count / total_samples if total_samples > 0 else 0
                stats_data.append([
                    state,
                    f"{count}",
                    f"{proportion:.4f}",
                    f"{proportion*100:.2f}%"
                ])
        
        if stats_data:
            table = ax2.table(cellText=stats_data,
                            colLabels=['State', 'Count', 'Proportion', 'Percentage'],
                            cellLoc='center',
                            loc='center',
                            bbox=[0, 0, 1, 1])
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 3)
            
            # Color code state names
            for i, row in enumerate(stats_data, 1):
                state = row[0]
                table[(i, 0)].set_facecolor(self.state_colors.get(state, '#95A5A6'))
                table[(i, 0)].set_text_props(color='white', weight='bold')
            
            # Style header
            for j in range(4):
                table[(0, j)].set_facecolor('#34495E')
                table[(0, j)].set_text_props(weight='bold', color='white')
        
        ax2.set_title('📊 State Summary Statistics', fontsize=14, fontweight='bold', pad=20)
        
        ax2.set_title('📊 State Summary Statistics', fontsize=14, fontweight='bold', pad=20)
        
        # ============ PLOT 3: Arousal/Valence Statistics Table ============
        ax3 = fig.add_subplot(gs[1, 2])
        ax3.axis('off')
        
        dimension_stats = []
        for dim in ['arousal', 'valence']:
            col_name = f'{prefix}{dim}'
            if col_name in df.columns:
                dimension_stats.append([
                    dim.capitalize(),
                    f"{df[col_name].mean():.4f}",
                    f"{df[col_name].std():.4f}",
                    f"{df[col_name].min():.4f}",
                    f"{df[col_name].max():.4f}"
                ])
        
        if dimension_stats:
            table2 = ax3.table(cellText=dimension_stats,
                             colLabels=['Dimension', 'Mean', 'Std Dev', 'Min', 'Max'],
                             cellLoc='center',
                             loc='center',
                             bbox=[0, 0, 1, 1])
            table2.auto_set_font_size(False)
            table2.set_fontsize(10)
            table2.scale(1, 3.5)
            
            # Color code dimension names
            table2[(1, 0)].set_facecolor(self.dimension_colors['arousal'])
            table2[(1, 0)].set_text_props(color='white', weight='bold')
            table2[(2, 0)].set_facecolor(self.dimension_colors['valence'])
            table2[(2, 0)].set_text_props(color='white', weight='bold')
            
            # Style header
            for j in range(5):
                table2[(0, j)].set_facecolor('#34495E')
                table2[(0, j)].set_text_props(weight='bold', color='white')
        
        ax3.set_title('📈 Dimension Statistics', fontsize=14, fontweight='bold', pad=20)
        
        # ============ PLOT 4: Average Emotional State Values (Bar Chart) ============
        ax4 = fig.add_subplot(gs[2, 0])
        
        if quadrant_col in df.columns:
            state_counts = df[quadrant_col].value_counts()
            state_avgs = state_counts / len(df)  # Convert to proportions
            
            colors = [self.state_colors.get(state, '#95A5A6') for state in state_avgs.index]
            bars = ax4.bar(range(len(state_avgs)), state_avgs.values,
                          color=colors, alpha=0.8, edgecolor='black', linewidth=2)
            ax4.set_xticks(range(len(state_avgs)))
            ax4.set_xticklabels(state_avgs.index, fontsize=12, fontweight='bold')
            ax4.set_ylabel('Average Proportion', fontsize=12)
            ax4.set_title('📊 Average Emotional State Values', fontsize=14, fontweight='bold')
            ax4.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for i, val in enumerate(state_avgs.values):
                ax4.text(i, val + 0.02, f'{val:.3f}\n({val*100:.1f}%)', 
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # ============ PLOT 5: Current Emotional State Over Time ============
        ax5 = fig.add_subplot(gs[2, 1:])
        
        if quadrant_col in df.columns:
            # Map states to numeric values for plotting
            state_mapping = {'Stressed': 3, 'Excited': 2, 'Calm': 1, 'Tired': 0}
            numeric_states = df[quadrant_col].map(state_mapping)
            
            # Plot with color segments
            for i in range(len(time_data)-1):
                state = df[quadrant_col].iloc[i]
                color = self.state_colors.get(state, '#95A5A6')
                ax5.plot(time_data.iloc[i:i+2], numeric_states.iloc[i:i+2],
                        color=color, linewidth=4, alpha=0.8)
            
            ax5.set_yticks([0, 1, 2, 3])
            ax5.set_yticklabels(['Tired', 'Calm', 'Excited', 'Stressed'], fontsize=11)
            ax5.set_xlabel('Time (seconds)', fontsize=12)
            ax5.set_title('📈 Emotional State Over Time (Color-Coded)', fontsize=14, fontweight='bold')
            ax5.grid(True, alpha=0.3, axis='x')
            
            # Add colored background regions
            ax5.axhspan(2.5, 3.5, facecolor=self.state_colors['Stressed'], alpha=0.1)
            ax5.axhspan(1.5, 2.5, facecolor=self.state_colors['Excited'], alpha=0.1)
            ax5.axhspan(0.5, 1.5, facecolor=self.state_colors['Calm'], alpha=0.1)
            ax5.axhspan(-0.5, 0.5, facecolor=self.state_colors['Tired'], alpha=0.1)
        
        # ============ PLOT 6: Emotional State Proportion (Pie Chart) ============
        ax6 = fig.add_subplot(gs[3, 0])
        
        if quadrant_col in df.columns:
            state_counts = df[quadrant_col].value_counts()
            colors_pie = [self.state_colors.get(state, '#95A5A6') for state in state_counts.index]
            labels_pie = [f'{state}\n{count/len(df)*100:.1f}%'
                         for state, count in state_counts.items()]
            
            ax6.pie(state_counts.values, labels=labels_pie,
                   colors=colors_pie, startangle=90,
                   textprops={'fontsize': 11, 'weight': 'bold'})
            
            ax6.set_title('🥧 State Proportion', fontsize=14, fontweight='bold')
        
        # ============ PLOT 7: Distribution Table ============
        ax7 = fig.add_subplot(gs[3, 1:])
        ax7.axis('off')
        
        if quadrant_col in df.columns:
            state_counts = df[quadrant_col].value_counts()
            
            # Include all states, even with 0 count
            table_data = []
            for state in ['Stressed', 'Excited', 'Tired', 'Calm']:
                count = state_counts.get(state, 0)
                proportion = count / len(df) if len(df) > 0 else 0
                percentage = proportion * 100
                table_data.append([
                    state,
                    f"{count}",
                    f"{proportion:.4f}",
                    f"{percentage:.2f}%"
                ])
            
            table3 = ax7.table(cellText=table_data,
                             colLabels=['State', 'Sample Count', 'Proportion', 'Percentage'],
                             cellLoc='center',
                             loc='center',
                             bbox=[0, 0, 1, 1])
            table3.auto_set_font_size(False)
            table3.set_fontsize(11)
            table3.scale(1, 3.5)
            
            # Color code state names
            for i, row in enumerate(table_data, 1):
                state = row[0]
                table3[(i, 0)].set_facecolor(self.state_colors.get(state, '#95A5A6'))
                table3[(i, 0)].set_text_props(color='white', weight='bold')
            
            # Style header
            for j in range(4):
                table3[(0, j)].set_facecolor('#34495E')
                table3[(0, j)].set_text_props(weight='bold', color='white')
            
            ax7.set_title('📊 State Distribution Table', fontsize=14, fontweight='bold', pad=20)
            
            # Add dominant state text at bottom
            dominant_state = state_counts.idxmax()
            dominant_count = state_counts.max()
            dominant_pct = dominant_count / len(df) * 100
            
            fig.text(0.5, 0.02,
                    f"Most Dominant State: {dominant_state} ({dominant_count} samples - {dominant_pct:.1f}%)",
                    ha='center', fontsize=14, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5',
                             facecolor=self.state_colors.get(dominant_state, '#95A5A6'),
                             edgecolor='black', linewidth=2))
        
        # Save
        save_path = f"{base_filename}_facial_states.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Facial States report saved: {save_path}")
        plt.close()
    
    def generate_all_reports(self, df, base_filename):
        """Generate all three facial reports"""
        print(f"\n{'='*80}")
        print("🎭 GENERATING FACIAL EMOTION REPORTS")
        print(f"{'='*80}\n")
        
        self.generate_emotion_report(df, base_filename)
        self.generate_dimensions_report(df, base_filename)
        self.generate_states_report(df, base_filename)
        
        print(f"\n{'='*80}")
        print("✅ ALL FACIAL REPORTS GENERATED SUCCESSFULLY!")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
        base_name = csv_file.replace('.csv', '').replace('_emotion_data', '')
    else:
        csv_file = "angry_emotion_data.csv"
        base_name = "angry"
    
    try:
        df = pd.read_csv(csv_file)
        generator = FacialReportGenerator()
        generator.generate_all_reports(df, base_name)
    except FileNotFoundError:
        print(f"❌ File not found: {csv_file}")
        print("Usage: python facial_reports.py [csv_file]")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

"""
Multimodal Emotion Fusion Module
Combines facial and voice emotion data using weighted post-fusion
Weights: Facial (70%) + Voice (30%)
"""
import pandas as pd
import numpy as np
import sys


class MultimodalEmotionFusion:
    """
    Fuses facial and voice emotion data with configurable weights
    Default: 70% facial, 30% voice (facial is more accurate)
    """
    
    def __init__(self, w_facial=0.7, w_voice=0.3):
        """
        Initialize fusion weights
        
        Args:
            w_facial: Weight for facial emotions (default 0.7)
            w_voice: Weight for voice emotions (default 0.3)
        """
        self.w_facial = w_facial
        self.w_voice = w_voice
        
        # Ensure weights sum to 1.0
        total = w_facial + w_voice
        self.w_facial = w_facial / total
        self.w_voice = w_voice / total
        
        print(f"💡 Fusion weights: Facial={self.w_facial:.1%}, Voice={self.w_voice:.1%}")
        
        # Emotion names
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    
    def fuse_emotions(self, df):
        """
        Fuse the 7 emotion probabilities using weighted average
        
        Args:
            df: DataFrame with facial_* and voice_* emotion columns
            
        Returns:
            DataFrame with fused_* emotion columns
        """
        print("🎭 Fusing emotion probabilities...")
        
        for emotion in self.emotions:
            facial_col = f'facial_{emotion}'
            voice_col = f'voice_{emotion}'
            fused_col = f'fused_{emotion}'
            
            if facial_col in df.columns and voice_col in df.columns:
                df[fused_col] = (
                    df[facial_col] * self.w_facial + 
                    df[voice_col] * self.w_voice
                )
            elif facial_col in df.columns:
                # If only facial available, use it
                df[fused_col] = df[facial_col]
                print(f"  ⚠️ {emotion}: Only facial data available")
            elif voice_col in df.columns:
                # If only voice available, use it
                df[fused_col] = df[voice_col]
                print(f"  ⚠️ {emotion}: Only voice data available")
        
        return df
    
    def fuse_dimensions(self, df):
        """
        Fuse arousal, valence, and intensity dimensions
        
        Args:
            df: DataFrame with facial_* and voice_* dimension columns
            
        Returns:
            DataFrame with fused_* dimension columns
        """
        print("📊 Fusing emotion dimensions...")
        
        dimensions = ['arousal', 'valence', 'intensity']
        
        for dim in dimensions:
            facial_col = f'facial_{dim}'
            voice_col = f'voice_{dim}'
            fused_col = f'fused_{dim}'
            
            if facial_col in df.columns and voice_col in df.columns:
                df[fused_col] = (
                    df[facial_col] * self.w_facial + 
                    df[voice_col] * self.w_voice
                )
                print(f"  ✓ {dim.capitalize()}: Combined")
            elif facial_col in df.columns:
                df[fused_col] = df[facial_col]
                print(f"  ⚠️ {dim.capitalize()}: Only facial available")
            elif voice_col in df.columns:
                df[fused_col] = df[voice_col]
                print(f"  ⚠️ {dim.capitalize()}: Only voice available")
        
        # Add stress from voice (facial doesn't have stress)
        if 'voice_stress' in df.columns:
            df['fused_stress'] = df['voice_stress']
            print(f"  ✓ Stress: From voice only")
        
        return df
    
    def determine_dominant_emotion(self, df):
        """
        Determine the dominant fused emotion at each timepoint
        
        Args:
            df: DataFrame with fused_* emotion columns
            
        Returns:
            DataFrame with fused_dominant_emotion column
        """
        print("🎯 Determining dominant fused emotions...")
        
        emotion_cols = [f'fused_{e}' for e in self.emotions if f'fused_{e}' in df.columns]
        
        if emotion_cols:
            df['fused_dominant_emotion'] = df[emotion_cols].idxmax(axis=1).str.replace('fused_', '')
        
        return df
    
    def calculate_quadrant(self, df):
        """
        Calculate emotional state quadrant from fused arousal and valence
        Stressed: arousal > 0, valence < 0
        Excited: arousal > 0, valence > 0
        Tired: arousal < 0, valence < 0
        Calm: arousal < 0, valence > 0
        
        Args:
            df: DataFrame with fused_arousal and fused_valence
            
        Returns:
            DataFrame with fused_quadrant column
        """
        print("🗺️ Calculating fused emotional states...")
        
        if 'fused_arousal' in df.columns and 'fused_valence' in df.columns:
            def get_quadrant(row):
                arousal = row['fused_arousal']
                valence = row['fused_valence']
                
                if arousal > 0 and valence > 0:
                    return 'Excited'
                elif arousal > 0 and valence < 0:
                    return 'Stressed'
                elif arousal < 0 and valence > 0:
                    return 'Calm'
                else:
                    return 'Tired'
            
            df['fused_quadrant'] = df.apply(get_quadrant, axis=1)
        
        return df
    
    def calculate_derived_metrics(self, df):
        """
        Calculate derived metrics from fused dimensions
        
        Args:
            df: DataFrame with fused dimensions
            
        Returns:
            DataFrame with additional derived columns
        """
        print("🔢 Calculating derived fused metrics...")
        
        # Positivity and negativity (based on valence)
        if 'fused_valence' in df.columns:
            df['fused_positivity'] = df['fused_valence']
            df['fused_negativity'] = -df['fused_valence']
        
        # Excitement and calmness (based on arousal)
        if 'fused_arousal' in df.columns:
            df['fused_excitement'] = df['fused_arousal'].clip(lower=0)  # Positive arousal
            df['fused_calmness'] = (-df['fused_arousal']).clip(lower=0)  # Negative arousal
        
        return df
    
    def analyze_agreement(self, df):
        """
        Analyze agreement between facial and voice modalities
        
        Args:
            df: DataFrame with facial_*, voice_*, and fused_* data
            
        Returns:
            DataFrame with agreement metrics
        """
        print("🤝 Analyzing modality agreement...")
        
        # Dominant emotion agreement
        if 'facial_dominant_emotion' in df.columns and 'voice_dominant_emotion' in df.columns:
            df['emotion_agreement'] = (
                df['facial_dominant_emotion'] == df['voice_dominant_emotion']
            ).astype(int)
            
            agreement_rate = df['emotion_agreement'].mean()
            print(f"  📊 Emotion agreement rate: {agreement_rate:.1%}")
        
        # Quadrant agreement
        if 'facial_quadrant' in df.columns and 'voice_quadrant' in df.columns:
            df['state_agreement'] = (
                df['facial_quadrant'] == df['voice_quadrant']
            ).astype(int)
            
            state_agreement_rate = df['state_agreement'].mean()
            print(f"  📊 State agreement rate: {state_agreement_rate:.1%}")
        
        # Dimension correlation
        for dim in ['arousal', 'valence', 'intensity']:
            facial_col = f'facial_{dim}'
            voice_col = f'voice_{dim}'
            
            if facial_col in df.columns and voice_col in df.columns:
                correlation = df[facial_col].corr(df[voice_col])
                print(f"  📈 {dim.capitalize()} correlation: {correlation:.3f}")
        
        return df
    
    def create_fusion_report(self, df, base_filename):
        """
        Create a summary report of the fusion process
        
        Args:
            df: Fused DataFrame
            base_filename: Base name for output files
        """
        print("\n" + "="*80)
        print("📋 FUSION SUMMARY REPORT")
        print("="*80)
        
        # Emotion distribution
        if 'fused_dominant_emotion' in df.columns:
            print("\n🎭 Fused Emotion Distribution:")
            emotion_counts = df['fused_dominant_emotion'].value_counts()
            for emotion, count in emotion_counts.items():
                pct = count / len(df) * 100
                print(f"  {emotion.capitalize():10s}: {count:4d} samples ({pct:5.1f}%)")
        
        # State distribution
        if 'fused_quadrant' in df.columns:
            print("\n🗺️ Fused Emotional State Distribution:")
            state_counts = df['fused_quadrant'].value_counts()
            for state, count in state_counts.items():
                pct = count / len(df) * 100
                print(f"  {state:10s}: {count:4d} samples ({pct:5.1f}%)")
        
        # Dimension statistics
        print("\n📊 Fused Dimension Statistics:")
        for dim in ['arousal', 'valence', 'intensity', 'stress']:
            col = f'fused_{dim}'
            if col in df.columns:
                print(f"  {dim.capitalize():10s}: Mean={df[col].mean():6.3f}, "
                      f"Std={df[col].std():6.3f}, Min={df[col].min():6.3f}, Max={df[col].max():6.3f}")
        
        # Agreement metrics
        if 'emotion_agreement' in df.columns:
            print("\n🤝 Modality Agreement:")
            print(f"  Emotion agreement: {df['emotion_agreement'].mean():.1%}")
        if 'state_agreement' in df.columns:
            print(f"  State agreement:   {df['state_agreement'].mean():.1%}")
        
        print("\n" + "="*80)
    
    def fuse_data(self, input_csv, output_csv=None):
        """
        Main fusion pipeline
        
        Args:
            input_csv: Path to input CSV with facial and voice data
            output_csv: Path to output CSV (auto-generated if None)
            
        Returns:
            Fused DataFrame
        """
        print("\n" + "="*80)
        print("🔮 MULTIMODAL EMOTION FUSION")
        print("="*80)
        print(f"📂 Input: {input_csv}")
        
        # Load data
        df = pd.read_csv(input_csv)
        print(f"✓ Loaded {len(df)} samples")
        
        # Perform fusion
        df = self.fuse_emotions(df)
        df = self.fuse_dimensions(df)
        df = self.determine_dominant_emotion(df)
        df = self.calculate_quadrant(df)
        df = self.calculate_derived_metrics(df)
        df = self.analyze_agreement(df)
        
        # Generate output filename if not provided
        if output_csv is None:
            base = input_csv.replace('_emotion_data.csv', '').replace('.csv', '')
            output_csv = f"{base}_fusion.csv"
        
        # Save fused data
        df.to_csv(output_csv, index=False)
        print(f"\n💾 Saved fused data: {output_csv}")
        
        # Create summary report
        base_filename = output_csv.replace('_fusion.csv', '').replace('.csv', '')
        self.create_fusion_report(df, base_filename)
        
        print("\n" + "="*80)
        print("✅ FUSION COMPLETE!")
        print("="*80 + "\n")
        
        return df


def main():
    """Command-line interface for fusion"""
    if len(sys.argv) > 1:
        input_csv = sys.argv[1]
    else:
        print("Usage: python fusion.py <input_csv>")
        print("Example: python fusion.py angry_emotion_data.csv")
        return
    
    # Create fusion processor with 70/30 weights
    fusion = MultimodalEmotionFusion(w_facial=0.7, w_voice=0.3)
    
    # Process data
    fused_df = fusion.fuse_data(input_csv)
    
    print("📊 Fused columns created:")
    fused_cols = [col for col in fused_df.columns if col.startswith('fused_')]
    for col in fused_cols:
        print(f"  - {col}")


if __name__ == "__main__":
    main()

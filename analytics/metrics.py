import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import os

class WormAnalytics:
    def __init__(self):
        self.reset_metrics()
        # Create analytics directory if it doesn't exist
        os.makedirs('analytics/reports', exist_ok=True)
        
    def reset_metrics(self):
        """Reset all metrics to empty lists"""
        self.metrics = {
            'episode': [],
            'timestamp': [],
            'avg_reward': [],
            'wall_collisions': [],
            'exploration_ratio': [],
            'movement_smoothness': [],
            'epsilon': []
        }
        
    def update_metrics(self, episode, metrics):
        """Update metrics with new values"""
        self.metrics['episode'].append(episode)
        self.metrics['timestamp'].append(datetime.now())
        self.metrics['avg_reward'].append(metrics.get('avg_reward', 0))
        self.metrics['wall_collisions'].append(metrics.get('wall_collisions', 0))
        self.metrics['exploration_ratio'].append(metrics.get('exploration_ratio', 0))
        self.metrics['movement_smoothness'].append(metrics.get('movement_smoothness', 0))
        self.metrics['epsilon'].append(metrics.get('epsilon', 0))
        
    def generate_report(self, episode_number):
        """Generate an HTML report with interactive plots"""
        df = pd.DataFrame(self.metrics)
        
        # Create an HTML report
        report_path = f'analytics/reports/report_episode_{episode_number}.html'
        
        # Create plotly figures
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Learning Progress',
                'Movement Metrics',
                'Exploration vs Epsilon',
                'Wall Collisions Over Time',
                'Movement Smoothness',
                'Exploration Coverage'
            )
        )

        # Learning Progress
        fig.add_trace(
            go.Scatter(
                name='Average Reward',
                y=df['avg_reward'],
                mode='lines',
                line=dict(color='green')
            ),
            row=1, col=1
        )

        # Movement Metrics
        fig.add_trace(
            go.Scatter(
                name='Movement Smoothness',
                y=df['movement_smoothness'],
                mode='lines',
                line=dict(color='blue')
            ),
            row=1, col=2
        )

        # Exploration vs Epsilon
        fig.add_trace(
            go.Scatter(
                name='Exploration Ratio',
                y=df['exploration_ratio'],
                mode='lines',
                line=dict(color='purple')
            ),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(
                name='Epsilon',
                y=df['epsilon'],
                mode='lines',
                line=dict(color='red')
            ),
            row=2, col=1
        )

        # Wall Collisions
        fig.add_trace(
            go.Scatter(
                name='Wall Collisions',
                y=df['wall_collisions'],
                mode='lines',
                line=dict(color='orange')
            ),
            row=2, col=2
        )

        # Movement Smoothness Distribution
        fig.add_trace(
            go.Histogram(
                name='Smoothness Distribution',
                x=df['movement_smoothness'],
                nbinsx=30,
                marker_color='blue'
            ),
            row=3, col=1
        )

        # Exploration Coverage
        fig.add_trace(
            go.Scatter(
                name='Exploration Coverage',
                y=df['exploration_ratio'].cummax(),
                mode='lines',
                line=dict(color='green')
            ),
            row=3, col=2
        )

        # Update layout
        fig.update_layout(
            height=1200,
            width=1000,
            title_text=f"Worm AI Analysis - Episode {episode_number}",
            showlegend=True
        )

        # Add annotations explaining metrics
        annotations = [
            dict(
                text="Higher rewards indicate better overall performance",
                x=0.2, y=-0.15,
                showarrow=False,
                xref='paper', yref='paper'
            ),
            dict(
                text="Smoother movement patterns are better",
                x=0.8, y=-0.15,
                showarrow=False,
                xref='paper', yref='paper'
            ),
            dict(
                text="Epsilon decreases as exploration reduces",
                x=0.2, y=-0.45,
                showarrow=False,
                xref='paper', yref='paper'
            )
        ]
        fig.update_layout(annotations=annotations)

        # Save the plot
        fig.write_html(report_path)

        # Generate summary statistics
        summary_stats = df.describe()
        summary_path = report_path.replace('.html', '_summary.html')
        summary_stats.to_html(summary_path)

        return report_path

    def plot_heatmap(self, positions_history, dimensions, episode_number):
        """Generate a heatmap of the worm's positions"""
        if not positions_history:
            return None
            
        # Extract x and y coordinates directly
        x = [pos[0] for pos in positions_history]
        y = [pos[1] for pos in positions_history]
        
        # Create heatmap using 2D histogram
        heatmap, xedges, yedges = np.histogram2d(
            x, y,
            bins=50,
            range=[[0, dimensions[0]], [0, dimensions[1]]]
        )
        
        # Create heatmap visualization
        plt.figure(figsize=(10, 10))
        plt.imshow(heatmap.T, origin='lower', extent=[0, dimensions[0], 0, dimensions[1]], 
                  cmap='viridis', interpolation='gaussian')
        plt.colorbar(label='Visit Frequency')
        plt.title(f'Worm Position Heatmap (Episode {episode_number})')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        
        # Save plot
        heatmap_path = f'analytics/reports/heatmap_episode_{episode_number}.png'
        plt.savefig(heatmap_path)
        plt.close()
        
        return heatmap_path

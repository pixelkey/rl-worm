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
            'wall_stays': [],
            'danger_zone_count': [],
            'exploration_ratio': [],
            'movement_smoothness': [],
            'epsilon': [],
            'deaths_by_starvation': [],  # Track deaths by starvation
            'deaths_by_wall': [],        # Track deaths by wall collision
            'death_penalty_total': []    # Track total death penalty per episode
        }
        
    def update_metrics(self, episode, metrics):
        """Update metrics with new values"""
        self.metrics['episode'].append(episode)
        self.metrics['timestamp'].append(datetime.now())
        for key in metrics:
            if key in self.metrics:
                self.metrics[key].append(metrics[key])
        
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

        print("\nTraining Report")
        print("-" * 50)
        
        # Calculate averages over last N episodes
        window = min(100, len(self.metrics['avg_reward']))
        
        # Reward metrics
        avg_reward = np.mean(self.metrics['avg_reward'][-window:])
        print(f"\nReward Metrics (last {window} episodes):")
        print(f"Average Reward: {avg_reward:.2f}")
        
        # Death statistics
        print("\nDeath Statistics:")
        if len(self.metrics['deaths_by_starvation']) > 0:
            starvation_deaths = np.sum(self.metrics['deaths_by_starvation'][-window:])
            wall_deaths = np.sum(self.metrics['deaths_by_wall'][-window:])
            total_deaths = starvation_deaths + wall_deaths
            print(f"Total Deaths: {total_deaths}")
            print(f"Deaths by Starvation: {starvation_deaths} ({(starvation_deaths/total_deaths*100):.1f}%)")
            print(f"Deaths by Wall Collision: {wall_deaths} ({(wall_deaths/total_deaths*100):.1f}%)")
            print(f"Average Death Penalty: {np.mean(self.metrics['death_penalty_total'][-window:]):.2f}")
        
        # Movement metrics
        print("\nMovement Metrics:")
        print(f"Wall Collisions: {np.mean(self.metrics['wall_collisions'][-window:]):.2f}")
        print(f"Wall Stays: {np.mean(self.metrics['wall_stays'][-window:]):.2f}")
        print(f"Danger Zone Count: {np.mean(self.metrics['danger_zone_count'][-window:]):.2f}")
        print(f"Movement Smoothness: {np.mean(self.metrics['movement_smoothness'][-window:]):.2f}")
        
        # Exploration metrics
        print("\nExploration Metrics:")
        print(f"Exploration Ratio: {np.mean(self.metrics['exploration_ratio'][-window:]):.2f}")
        print(f"Current Epsilon: {self.metrics['epsilon'][-1]:.3f}")
        
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

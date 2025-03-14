import numpy as np
import matplotlib.pyplot as plt
import os

class EMATracker:
    def __init__(self, beta=0.9, 
                 save_dir=None,
                 name=None):
        """
        Initialize the EMA tracker with a beta value.
        
        Args:
            beta (float): The smoothing factor (between 0 and 1).
                          Higher values give more weight to past observations.
        """
        self.beta = beta
        self.ema = None
        self.loss_steps = []
        self.score_steps = []
        self.loss_history = []
        self.ema_loss_history = []
        self.score_history = []
        
        self.name = name
        if save_dir is not None and name is not None:
            self.save_path = os.path.join(save_dir, f"{name}.pkl")
        
        if os.path.exists(self.save_path):
            self.load(load_path=self.save_path)
        
    def update(self, step, loss, score=None):
        """
        Update the EMA with a new loss value.
        
        Args:
            loss (float): The new loss value to incorporate.
            
        Returns:
            float: The updated EMA value.
        """
        # Store raw loss
        self.loss_history.append(loss)
        self.loss_steps.append(step)
        
        # Update EMA
        if self.ema is None:
            self.ema = loss  # Initialize EMA with first loss value
        else:
            self.ema = self.beta * self.ema + (1 - self.beta) * loss
        
        # Store EMA
        self.ema_loss_history.append(self.ema)
        
        if score is not None:
            self.score_history.append(score)
            self.score_steps.append(step)
        
        return self.ema
    
    def update_score(self, step, score):
        self.score_history.append(score)
        self.score_steps.append(step)
    
    def save(self):
        import pickle
        if self.save_path is not None:
            stats_dict = {
                "loss_steps": self.loss_steps,
                "loss_history": self.loss_history,
                "ema_loss_history": self.ema_loss_history,
                "score_steps": self.score_steps,
                "score_history": self.score_history
            }
            with open(self.save_path, "wb") as trg:
                pickle.dump(stats_dict, trg)
    
    def load(self, load_path: str):
        import pickle
        with open(load_path, "rb") as trg:
            stats_dict = pickle.load(trg)
        
        self.loss_steps = stats_dict["loss_steps"]
        self.loss_history = stats_dict["loss_history"]
        self.ema_loss_history = stats_dict["ema_loss_history"]
        self.score_steps = stats_dict["score_steps"]
        self.score_history = stats_dict["score_history"]
        print(f"Loaded losses from step [{self.loss_steps[-1]}]")
        print(f"Loaded scores from step [{self.score_steps[-1]}]")
        
    def plot(self):
        """Plot the raw loss values and the EMA smoothed values."""
        # iterations = range(1, len(self.loss_history) + 1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_steps, self.loss_history, 'b-', alpha=0.5, label='Raw Loss')
        plt.plot(self.loss_steps, self.ema_loss_history, 'r-', linewidth=2, label=f'EMA (β={self.beta})')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Domain Loss and EMA')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

# Example usage
if __name__ == "__main__":
    # Create a tracker with beta=0.9
    tracker = EMATracker(beta=0.9,
                         save_dir="/scratch/homes/sfan/test",
                         name="test_domain")
    
    # Simulate some domain loss values (e.g., from training)
    # Replace this with your actual loss values
    np.random.seed(42)
    losses = np.random.normal(loc=5.0, scale=1.0, size=100)
    losses = np.exp(-np.linspace(0, 3, 100)) * 5 + losses * 0.5  # Trend + noise
    
    # Track losses and get EMA values
    ema_values = [tracker.update(i,loss,score=loss) for i,loss in enumerate(losses)]
    
    # Plot the results
    tracker.plot()
    tracker.save()
    
    # Print some statistics
    print(f"Final raw loss: {losses[-1]:.4f}")
    print(f"Final EMA loss (β={tracker.beta}): {ema_values[-1]:.4f}")
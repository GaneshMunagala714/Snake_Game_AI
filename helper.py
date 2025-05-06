import matplotlib.pyplot as plt

def plot(scores, mean_scores):
    plt.clf()
    plt.title('Training Performance')
    plt.xlabel('Game')
    plt.ylabel('Score')
    plt.plot(scores, label='Score')
    plt.plot(mean_scores, label='Mean Score')
    plt.legend()
    plt.pause(0.1)

import matplotlib.pyplot as plt

# Replace these with your actual results from evaluation logs
categories = ['Retrieve', 'isREL', 'isSUP', 'isUSE']
accuracies = [80.00, 80.00, 70.00, 55.00]

plt.figure(figsize=(8,5))
plt.bar(categories, accuracies)
plt.title('Critic Model Accuracy by Category')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 100)
plt.grid(axis='y')
plt.show()

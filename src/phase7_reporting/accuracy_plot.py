import matplotlib.pyplot as plt


def plot_accuracy(before, after):

    models = ["Before Cleaning", "After Cleaning"]
    accuracy = [before, after]

    plt.figure()

    plt.bar(models, accuracy)

    plt.title("ML Accuracy Comparison")

    plt.ylabel("Accuracy")

    plt.savefig("accuracy_comparison.png")

    plt.show()
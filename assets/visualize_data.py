# Visualize a couple datapoints, pulled randomly from the training set.
import matplotlib.pyplot as plt

shuffled_indices = torch.randperm(train_dataset.__len__()).tolist()
random_samples = [train_dataset[x] for x in shuffled_indices]

fig, axs = plt.subplots(5, 5, figsize=(15, 15))
for i, ax in enumerate(axs.flat):
    data, target = random_samples[i]

    image = data.cpu().numpy()
    target = int(target.cpu().numpy())
    image = image.transpose(1, 2, 0)

    ax.set_title(int_to_class[target], fontsize=10)
    ax.imshow(image)
    ax.axis("off")

plt.show()

# Also: data exploration, data statistics, etc.
